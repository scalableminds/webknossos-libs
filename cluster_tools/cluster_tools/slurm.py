"""Abstracts access to a Slurm cluster via its command-line tools.
"""
import re
import os
import threading
import time
from .util import chcall, random_string, local_filename

LOG_FILE = local_filename("slurmpy.log")
OUTFILE_FMT = local_filename("slurmpy.stdout.{}.log")


def submit_text(job, job_name):
    """Submits a Slurm job represented as a job file string. Returns
    the job ID.
    """
    job_name_arg = ""
    if job_name is not None:
        job_name_arg = '--job-name "{}"'.format(job_name)

    filename = local_filename("_temp_{}.sh".format(random_string()))
    with open(filename, "w") as f:
        f.write(job)
    jobid, _ = chcall("sbatch {} --parsable {}".format(job_name_arg, filename))
    os.unlink(filename)
    return int(jobid)


def submit(
    cmdline,
    outpath_fmt=OUTFILE_FMT,
    job_resources=None,
    job_name=None,
    additional_setup_lines=[],
    job_count=None,
):
    """Starts a Slurm job that runs the specified shell command line.
    """

    outpath = outpath_fmt.format("%j" if job_count is None else "%A.%a")

    job_resources_lines = []
    if job_resources is not None:
        for resource, value in job_resources.items():
            job_resources_lines += ["#SBATCH --{}={}".format(resource, value)]

    job_array_line = ""
    if job_count is not None:
        job_array_line = "#SBATCH --array=0-{}".format(job_count - 1)

    script_lines = (
        ["#!/bin/sh", "#SBATCH --output={}".format(outpath), job_array_line]
        + job_resources_lines
        + [*additional_setup_lines, "srun {}".format(cmdline)]
    )

    return submit_text("\n".join(script_lines), job_name)
