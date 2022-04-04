"""Abstracts access to a Kubernetes cluster via its Python library."""
import concurrent
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import uuid4

import kubernetes
import kubernetes.client.models as kubernetes_models
from typing_extensions import Literal

from .cluster_executor import ClusterExecutor


def volume_name_from_path(path: Path) -> str:
    return f"{(hash(str(path)) & sys.maxsize):016x}"


def deduplicate_mounts(mounts: List[Path]) -> List[Path]:
    output = []
    unique_mounts = set(mounts)
    for mount in unique_mounts:
        if not any(m in mount.parents for m in unique_mounts):
            output.append(mount)
    return output


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

    @classmethod
    def executor_key(cls) -> str:
        return "kubernetes"

    @staticmethod
    def format_log_file_name(job_id_with_index: str, suffix=".stdout") -> str:
        return "kube.{}.log{}".format(str(job_id_with_index), suffix)

    @staticmethod
    def get_job_array_index() -> Optional[int]:
        # In kubernetes all jobs are array jobs. `JOB_COMPLETION_INDEX` is set by
        # kubernetes directly. Both variables should exist (if it is a kubernetes
        # job), but `JOB_IS_ARRAY_JOB` can be `False`.
        if (
            "JOB_IS_ARRAY_JOB" in os.environ
            and "JOB_COMPLETION_INDEX" in os.environ
            and os.environ["JOB_IS_ARRAY_JOB"] == "True"
        ):
            return int(os.environ["JOB_COMPLETION_INDEX"])
        return None

    @staticmethod
    def get_current_job_id() -> Optional[str]:
        return os.environ.get("JOB_ID", None)

    @classmethod
    def get_job_id_string(cls) -> Optional[str]:
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
                body=kubernetes_models.V1Namespace(
                    api_version="v1",
                    kind="Namespace",
                    metadata=kubernetes_models.V1ObjectMeta(
                        name=self.job_resources["namespace"]
                    ),
                )
            )

    def get_python_executable(self):
        return self.job_resources.get("python_executable", "python")

    def inner_submit(
        self,
        cmdline: str,
        job_name: Optional[str] = None,
        additional_setup_lines: Optional[List[str]] = None,
        job_count: Optional[int] = None,
    ) -> Tuple[List["concurrent.futures.Future[str]"], List[Tuple[int, int]]]:
        """Starts a Kubernetes pod that runs the specified shell command line."""

        kubernetes_client = KubernetesClient()
        self.ensure_kubernetes_namespace()
        job_id = str(uuid4())

        job_id_future: "concurrent.futures.Future[str]" = concurrent.futures.Future()
        job_id_future.set_result(job_id)
        job_id_futures = [job_id_future]

        is_array_job = job_count is not None
        number_of_subjobs = job_count if job_count is not None else 1
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
        mounts = deduplicate_mounts(
            [Path(mount) for mount in self.job_resources["mounts"]]
            + [Path.cwd(), Path(self.cfut_dir).absolute()]
        )

        job_manifest = kubernetes_models.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=kubernetes_models.V1ObjectMeta(name=job_id),
            spec=kubernetes_models.V1JobSpec(
                completion_mode="Indexed",
                completions=number_of_subjobs,
                parallelism=number_of_subjobs,
                ttl_seconds_after_finished=604800,  # 7 days
                template=kubernetes_models.V1PodTemplateSpec(
                    metadata=kubernetes_models.V1ObjectMeta(
                        annotations={
                            "cluster-tools.scalableminds.com/job-id": job_id,
                            "cluster-tools.scalableminds.com/job-is-array-job": str(
                                is_array_job
                            ),
                            "cluster-tools.scalableminds.com/job-name": job_name
                            if job_name is not None
                            else "",
                        }
                    ),
                    spec=kubernetes_models.V1PodSpec(
                        containers=[
                            kubernetes_models.V1Container(
                                image=self.job_resources["image"],
                                image_pull_policy="IfNotPresent",
                                working_dir=str(Path.cwd().absolute()),
                                command=["/bin/bash"],
                                name="worker",
                                args=[
                                    "-c",
                                    f"{umaskline}{cmdline} 0 2>&1 > >(tee -a {log_path})",
                                ],
                                env=[
                                    kubernetes_models.V1EnvVar(name=name, value=value)
                                    for name, value in os.environ.items()
                                    if name not in ("PWD", "OLDPWD")
                                    and re.match("^[-._a-zA-Z][-._a-zA-Z0-9]*$", name)
                                    is not None
                                ]
                                + [
                                    kubernetes_models.V1EnvVar(
                                        name="JOB_ID", value=job_id
                                    ),
                                    kubernetes_models.V1EnvVar(
                                        name="JOB_IS_ARRAY_JOB", value=str(is_array_job)
                                    ),
                                ],
                                security_context=kubernetes_models.V1SecurityContext(
                                    run_as_user=os.getuid(), run_as_group=os.getgid()
                                ),
                                resources=kubernetes_models.V1ResourceRequirements(
                                    requests=requested_resources
                                ),
                                volume_mounts=[
                                    kubernetes_models.V1VolumeMount(
                                        name=volume_name_from_path(mount),
                                        mount_path=str(mount),
                                    )
                                    for mount in mounts
                                ],
                            )
                        ],
                        node_selector=self.job_resources.get("node_selector"),
                        restart_policy="Never",
                        volumes=[
                            kubernetes_models.V1Volume(
                                name=volume_name_from_path(mount),
                                host_path=kubernetes_models.V1HostPathVolumeSource(
                                    path=str(mount)
                                ),
                            )
                            for mount in mounts
                        ],
                    ),
                ),
            ),
        )

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
    ) -> Literal["failed", "ignore", "completed"]:
        kubernetes_client = KubernetesClient()
        [job_id, job_index] = (
            job_id_with_index.split("_")
            if "_" in job_id_with_index
            else [job_id_with_index, "0"]
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
                        is_array_job = (
                            pod.metadata.annotations[
                                "cluster-tools.scalableminds.com/job-is-array-job"
                            ]
                            == "True"
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
