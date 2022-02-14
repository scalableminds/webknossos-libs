"""Abstracts access to a Slurm cluster via its command-line tools.
"""
import concurrent
import os
from typing import List, Optional, Union
from uuid import uuid4

import kubernetes

from .cluster_executor import ClusterExecutor


class KubernetesExecutor(ClusterExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.job_resources is None:
            self.job_resources = {}
        if "namespace" not in self.job_resources:
            self.job_resources["namespace"] = "default"
        if "node_selector" not in self.job_resources:
            self.job_resources["node_selector"] = {}
        if "image" not in self.job_resources:
            self.job_resources["image"] = "scalableminds/webknossos-cuber:latest"

    @staticmethod
    def format_log_file_name(jobid: str, suffix=".stdout") -> str:
        return "kube.{}.log{}".format(str(jobid), suffix)

    @staticmethod
    def get_job_array_index() -> Optional[str]:
        return os.environ.get("CLUSTER_JOB_INDEX", None)

    @staticmethod
    def get_current_job_id() -> Optional[str]:
        return os.environ.get("CLUSTER_JOB_ID")

    @classmethod
    def get_job_id_string(cls) -> Optional[str]:
        return cls.get_current_job_id()

    def get_jobid_with_index(self, jobid: str, index: int) -> str:
        return f"{jobid}--{index}"

    def get_kubernetes_client(self):
        kubernetes.config.load_kube_config()
        core_v1 = kubernetes.client.api.core_v1_api.CoreV1Api()
        return core_v1

    def get_python_executable(self):
        return self.job_resources.get("python_executable", "python")

    def inner_submit(
        self, cmdline: str, job_name: Optional[str] = None, job_count: int = 1, **_
    ):
        """Starts a Kubernetes pod that runs the specified shell command line."""

        kubernetes_client = self.get_kubernetes_client()
        array_job_id = uuid4()

        job_id_future = concurrent.futures.Future()
        job_id_future.set_result(array_job_id)
        job_id_futures = [job_id_future]
        ranges = [(0, job_count)]

        requested_resources = {
            k: v
            for k, v in self.job_resources.items()
            if k in ("memory", "cpu") or k.startswith("hugepages-")
        }

        for job_index in range(0, job_count):
            pod_name = f"{array_job_id}--{job_index}"
            stdout_path = self.format_log_file_path(self.cfut_dir, pod_name)
            stderr_path = self.format_log_file_path(
                self.cfut_dir, pod_name, suffix=".stderr"
            )
            pod_manifest = {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": pod_name,
                    "annotations": {
                        "cluster-tools.scalableminds.com/job_id": array_job_id,
                        "cluster-tools.scalableminds.com/job_index": job_index,
                        "cluster-tools.scalableminds.com/job_name": job_name,
                    },
                },
                "spec": {
                    "containers": [
                        {
                            "image": self.job_resources["image"],
                            "imagePullPolicy": "IfNotPresent",
                            "name": pod_name,
                            "workingDir": os.path.abspath(os.path.curdir),
                            "command": ["/bin/bash"],
                            "args": [
                                "-c",
                                f"{cmdline} 0 > >(tee -a {stdout_path}) 2> >(tee -a {stderr_path} >&2)",
                            ],
                            "env": [
                                {"name": name, "value": value}
                                for name, value in os.environ.items()
                                if name not in ("PWD", "OLDPWD")
                            ]
                            + [
                                {"name": "CLUSTER_JOB_ID", "value": pod_name},
                                {"name": "CLUSTER_JOB_INDEX", "value": str(job_index)},
                            ],
                            "resources": {"requests": requested_resources},
                            "volumeMounts": [
                                # {
                                #     "name": "cfut-dir",
                                #     "mountPath": os.path.abspath(self.cfut_dir),
                                # },
                                {"name": "srv", "mountPath": "/srv"},
                                {
                                    "name": "cwd",
                                    "mountPath": os.path.abspath(os.path.curdir),
                                },
                            ],
                        }
                    ],
                    "nodeSelector": self.job_resources.get("node_selector"),
                    "restartPolicy": "Never",
                    "volumes": [
                        # {
                        #     "name": "cfut-dir",
                        #     "hostPath": {"path": os.path.abspath(self.cfut_dir)},
                        # },
                        {"name": "srv", "hostPath": {"path": "/srv"}},
                        {
                            "name": "cwd",
                            "hostPath": {"path": os.path.abspath(os.path.curdir)},
                        },
                    ],
                },
            }
            try:
                kubernetes_client.create_namespaced_pod(
                    body=pod_manifest, namespace=self.job_resources["namespace"]
                )
            except Exception as e:
                print(e)

        return job_id_futures, ranges

    def check_for_crashed_job(
        self, job_id: str
    ) -> Union["failed", "ignore", "completed"]:
        kubernetes_client = self.get_kubernetes_client()
        resp = kubernetes_client.read_namespaced_pod(
            name=job_id, namespace=self.job_resources["namespace"]
        )
        if resp.status.phase == "Failed":
            return "failed"
        if resp.status.phase == "Succeeded":
            return "completed"
        return "ignore"

    def get_number_of_submitted_jobs(self) -> int:
        kubernetes_client = self.get_kubernetes_client()
        resp = kubernetes_client.list_namespaced_pod(
            namespace=self.job_resources["namespace"]
        )
        return len(resp.items)

    def get_pending_tasks(self) -> List[str]:
        kubernetes_client = self.get_kubernetes_client()
        resp = kubernetes_client.list_namespaced_pod(
            namespace=self.job_resources["namespace"]
        )
        return [
            pod.metadata.name
            for pod in resp.items
            if pod.status.phase in ("Pending", "Running", "Unknown")
        ]
