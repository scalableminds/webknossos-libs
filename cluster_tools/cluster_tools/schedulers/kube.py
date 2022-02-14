"""Abstracts access to a Slurm cluster via its command-line tools.
"""
import concurrent
import logging
import os
import re
from uuid import uuid4
import kubernetes
import sys
import threading
from functools import lru_cache
from typing import Union

from cluster_tools.util import call, chcall, random_string

from .cluster_executor import ClusterExecutor


SLURM_QUEUE_CHECK_INTERVAL = 1 if "pytest" in sys.modules else 60


def noopDecorator(func):
    return func


cache_in_production = noopDecorator if "pytest" in sys.modules else lru_cache(maxsize=1)


class KubernetesExecutor(ClusterExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kubernetes_namespace = "default"
        self.kubernetes_node_selector = {"kubernetes.io/hostname": "asterix.scm.io"}
        self.image = "scalableminds/webknossos-cuber:v0.9.5"

    @staticmethod
    def format_log_file_name(job_id, suffix=".stdout"):
        return "kube.{}.log{}".format(str(job_id), suffix)

    @staticmethod
    def get_current_job_id():
        return os.environ.get("CLUSTER_JOB_ID")

    @classmethod
    def get_job_id_string(cls):
        return cls.get_current_job_id()

    def handle_kill(self, *args, **kwargs):
        # TODO
        super().handle_kill(*args, **kwargs)

    @staticmethod
    def get_kubernetes_client():
        kubernetes.config.load_kube_config()
        core_v1 = kubernetes.client.api.core_v1_api.CoreV1Api()
        return core_v1

    def inner_submit(
        self, cmdline, job_name=None, additional_setup_lines=None, job_count=1
    ):
        """Starts a Slurm job that runs the specified shell command line."""
        if additional_setup_lines is None:
            additional_setup_lines = []

        kubernetes_client = KubernetesExecutor.get_kubernetes_client()
        array_job_id = uuid4()

        job_id_futures = []
        ranges = [(0, job_count)]

        for job_index in range(0, job_count):
            pod_name = f"{array_job_id}_{job_index}"
            stdout_path = self.format_log_file_path(self.cfut_dir, pod_name)
            stderr_path = self.format_log_file_path(self.cfut_dir, pod_name)
            pod_manifest = {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {"name": pod_name},
                "spec": {
                    "containers": [
                        {
                            "image": self.image,
                            "imagePullPolicy": "IfNotPresent",
                            "name": pod_name,
                            "workingDir": os.path.curdir,
                            "args": [
                                "/bin/sh",
                                "-c",
                                f'"{cmdline}" "{job_index}" > {stdout_path} 2> {stderr_path}',
                            ],
                            "env": [
                                {"name": name, "value": value}
                                for name, value in os.environ.items()
                            ]
                            + [
                                {"name": "CLUSTER_JOB_ID", "value": pod_name},
                                {"name": "CLUSTER_JOB_INDEX", "value": str(job_index)},
                            ],
                            "resources": {
                                "requests": self.job_resources
                                if self.job_resources is not None
                                else {}
                            },
                            "volumeMounts": [
                                {
                                    "name": "cfut_dir",
                                    "mountPath": os.path.abspath(self.cfut_dir),
                                },
                                {"name": "srv", "mountPath": "/srv"},
                                {"name": "cwd", "mountPath": os.path.curdir},
                            ],
                        }
                    ],
                    "nodeSelector": self.kubernetes_node_selector,
                    "lifecycle": {"restartPolicy": "Never"},
                    "volumes": [
                        {
                            "name": "cfut_dir",
                            "hostPath": {"path": os.path.abspath(self.cfut_dir)},
                        },
                        {"name": "srv", "hostPath": {"path": "/srv"}},
                        {"name": "cwd", "hostPath": {"path": os.path.curdir}},
                    ],
                },
            }
            resp = kubernetes_client.create_namespaced_pod(
                body=pod_manifest, namespace=self.kubernetes_namespace
            )
            print(resp)

            job_id_future = concurrent.futures.Future()
            job_id_future.set_result(pod_name)
            job_id_futures.append(job_id_future)

        return job_id_futures, ranges

    def check_for_crashed_job(self, job_id) -> Union["failed", "ignore", "completed"]:
        kubernetes_client = KubernetesExecutor.get_kubernetes_client()
        resp = kubernetes_client.read_namespaced_pod(
            name=job_id, namespace=self.kubernetes_namespace
        )
        print(resp)
        if resp.status.phase == "Failed":
            return "failed"
        if resp.status.phase == "Succeeded":
            return "completed"
        return "ignore"

    def get_number_of_submitted_jobs(self):
        kubernetes_client = KubernetesExecutor.get_kubernetes_client()
        resp = kubernetes_client.list_namespaced_pod(
            namespace=self.kubernetes_namespace
        )
        return len(resp)

    def get_pending_tasks(self):
        kubernetes_client = KubernetesExecutor.get_kubernetes_client()
        resp = kubernetes_client.list_namespaced_pod(
            namespace=self.kubernetes_namespace
        )
        print(resp)
        return [
            pod.spec.metadata.name for pod in resp if pod.status.phase != "Completed"
        ]
