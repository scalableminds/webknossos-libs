"""Abstracts access to a Kubernetes cluster via its Python library."""
import concurrent
import os
import re
import sys
from typing import List, Optional, Union
from uuid import uuid4

import kubernetes

from .cluster_executor import ClusterExecutor


def volume_name_from_path(path: str) -> str:
    return f"{(hash(path) & sys.maxsize):016x}"


class KubernetesClient:
    def __init__(self):
        kubernetes.config.load_kube_config()
        self.core = kubernetes.client.api.core_v1_api.CoreV1Api()
        self.batch = kubernetes.client.api.batch_v1_api.BatchV1Api()


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
        if "mounts" not in self.job_resources:
            self.job_resources["mounts"] = []

    @staticmethod
    def format_log_file_name(job_id_with_index: str, suffix=".stdout") -> str:
        return "kube.{}.log{}".format(str(job_id_with_index), suffix)

    @staticmethod
    def get_job_array_index() -> Optional[int]:
        if bool(os.environ["JOB_IS_ARRAY_JOB"]):
            return int(os.environ["JOB_COMPLETION_INDEX"])
        return None

    @staticmethod
    def get_current_job_id() -> str:
        return os.environ["JOB_ID"]

    @classmethod
    def get_job_id_string(cls) -> str:
        job_id = cls.get_current_job_id()
        job_index = cls.get_job_array_index()
        if job_index is None:
            return job_id
        return cls.get_jobid_with_index(job_id, job_index)

    def ensure_kubernetes_namespace(self):
        kubernetes_client = KubernetesClient()
        try:
            kubernetes_client.core.read_namespace(self.job_resources["namespace"])
        except kubernetes.client.exceptions.ApiException as e:
            if e.status != 404:
                raise e
            kubernetes_client.core.create_namespace(
                body={
                    "apiVersion": "v1",
                    "kind": "Namespace",
                    "metadata": {
                        "name": self.job_resources["namespace"],
                    },
                }
            )

    def get_python_executable(self):
        return self.job_resources.get("python_executable", "python")

    def inner_submit(
        self,
        cmdline: str,
        job_name: Optional[str] = None,
        job_count: Optional[int] = None,
        **_,
    ):
        """Starts a Kubernetes pod that runs the specified shell command line."""

        kubernetes_client = KubernetesClient()
        self.ensure_kubernetes_namespace()
        job_id = str(uuid4())

        job_id_future = concurrent.futures.Future()
        job_id_future.set_result(job_id)
        job_id_futures = [job_id_future]

        is_array_job = job_count is not None
        number_of_subjobs = job_count if is_array_job else 1
        ranges = [(0, number_of_subjobs)]

        requested_resources = {
            k: v
            for k, v in self.job_resources.items()
            if k in ("memory", "cpu") or k.startswith("hugepages-")
        }
        umaskline = (
            f"umask {self.job_resources['umask']}; "
            if "umask" in self.job_resources
            else ""
        )
        log_path = (
            self.format_log_file_path(self.cfut_dir, f"{job_id}_$JOB_COMPLETION_INDEX")
            if is_array_job
            else self.format_log_file_path(self.cfut_dir, job_id)
        )

        job_manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_id,
            },
            "spec": {
                "completionMode": "Indexed",
                "completions": number_of_subjobs,
                "parallelism": number_of_subjobs,
                "ttlSecondsAfterFinished": 604800,  # 7 days
                "template": {
                    "metadata": {
                        "annotations": {
                            "cluster-tools.scalableminds.com/job-id": job_id,
                            "cluster-tools.scalableminds.com/job-is-array-job": str(
                                is_array_job
                            ),
                            "cluster-tools.scalableminds.com/job-name": job_name
                            if job_name is not None
                            else "",
                        },
                    },
                    "spec": {
                        "containers": [
                            {
                                "image": self.job_resources["image"],
                                "imagePullPolicy": "IfNotPresent",
                                "workingDir": os.path.abspath(os.path.curdir),
                                "command": ["/bin/bash"],
                                "name": "worker",
                                "args": [
                                    "-c",
                                    f"{umaskline}{cmdline} 0 2>&1 > >(tee -a {log_path})",
                                ],
                                "env": [
                                    {"name": name, "value": value}
                                    for name, value in os.environ.items()
                                    if name not in ("PWD", "OLDPWD")
                                    and re.match("^[-._a-zA-Z][-._a-zA-Z0-9]*$", name)
                                    is not None
                                ]
                                + [
                                    {"name": "JOB_ID", "value": job_id},
                                    {
                                        "name": "JOB_IS_ARRAY_JOB",
                                        "value": str(is_array_job),
                                    },
                                ],
                                "securityContext": {
                                    "runAsUser": os.getuid(),
                                    "runAsGroup": os.getgid(),
                                },
                                "resources": {"requests": requested_resources},
                                "volumeMounts": [
                                    {
                                        "name": volume_name_from_path(mount),
                                        "mountPath": mount,
                                    }
                                    for mount in self.job_resources["mounts"]
                                ]
                                + [
                                    {
                                        "name": "cwd",
                                        "mountPath": os.path.abspath(os.curdir),
                                    },
                                    {
                                        "name": "cfut-dir",
                                        "mountPath": os.path.abspath(self.cfut_dir),
                                    },
                                ],
                            }
                        ],
                        "nodeSelector": self.job_resources.get("node_selector"),
                        "restartPolicy": "Never",
                        "volumes": [
                            {
                                "name": volume_name_from_path(mount),
                                "hostPath": {"path": mount},
                            }
                            for mount in self.job_resources["mounts"]
                        ]
                        + [
                            {
                                "name": "cwd",
                                "hostPath": {"path": os.path.abspath(os.curdir)},
                            },
                            {
                                "name": "cfut-dir",
                                "hostPath": {"path": os.path.abspath(self.cfut_dir)},
                            },
                        ],
                    },
                },
            },
        }
        try:
            kubernetes_client.batch.create_namespaced_job(
                body=job_manifest, namespace=self.job_resources["namespace"]
            )
        except Exception as e:
            print(e, type(e))
            raise e

        return job_id_futures, ranges

    def check_for_crashed_job(
        self, job_id_with_index: str
    ) -> Union["failed", "ignore", "completed"]:
        kubernetes_client = KubernetesClient()
        [job_id, job_index] = (
            job_id_with_index.split("_") if "_" in job_id else [job_id, 0]
        )
        resp = kubernetes_client.core.list_namespaced_pod(
            namespace=self.job_resources["namespace"],
            label_selector=f"job-name={job_id}",
        )
        for pod in resp.items:
            if (
                pod.metadata.annotations["batch.kubernetes.io/job-completion-index"]
                == job_index
            ):
                if pod.status.phase == "Failed":
                    return "failed"
                if pod.status.phase == "Succeeded":
                    return "completed"
                return "ignore"
        return "ignore"

    def get_number_of_submitted_jobs(self) -> int:
        kubernetes_client = KubernetesClient()
        resp = kubernetes_client.batch.list_namespaced_job(
            namespace=self.job_resources["namespace"]
        )
        return sum(
            (0 if job.status.active is None else job.status.active)
            + (0 if job.status.failed is None else job.status.failed)
            + (0 if job.status.succeeded is None else job.status.succeeded)
            for job in resp.items
        )

    def get_pending_tasks(self) -> List[str]:
        kubernetes_client = KubernetesClient()
        resp = kubernetes_client.batch.list_namespaced_job(
            namespace=self.job_resources["namespace"]
        )
        output = []
        for job in resp.items:
            if job.status.active is not None and job.status.active > 0:
                resp = kubernetes_client.core.list_namespaced_pod(
                    namespace=self.job_resources["namespace"],
                    label_selector=f"job-name={job.metadata.name}",
                )
                for pod in resp.items:
                    if pod.status.phase in ("Pending", "Running", "Unknown"):
                        job_id = pod.metadata.annotations[
                            "cluster-tools.scalableminds.com/job-id"
                        ]
                        is_array_job = bool(
                            pod.metadata.annotations[
                                "cluster-tools.scalableminds.com/job-is-array-job"
                            ]
                        )
                        if is_array_job:
                            job_index = int(
                                pod.metadata.annotations[
                                    "batch.kubernetes.io/job-completion-index"
                                ]
                            )
                            output.append(
                                self.get_jobid_with_index(
                                    job_id,
                                    job_index,
                                )
                            )
                        else:
                            output.append(job_id)
        return output
