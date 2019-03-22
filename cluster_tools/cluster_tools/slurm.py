"""Abstracts access to a Slurm cluster via its command-line tools.
"""
import re
import os
import threading
import time
from .util import chcall, random_string, local_filename

LOG_FILE = local_filename("slurmpy.log")
OUTFILE_FMT = local_filename("slurmpy.stdout.{}.log")

def submit_text(job):
    """Submits a Slurm job represented as a job file string. Returns
    the job ID.
    """
    filename = local_filename('_temp_{}.sh'.format(random_string()))
    with open(filename, 'w') as f:
        f.write(job)
    jobid, _ = chcall('sbatch --parsable {}'.format(filename))
    os.unlink(filename)
    return int(jobid)

def submit(cmdline, outpath=OUTFILE_FMT.format('%j'), additional_setup_lines=[], job_resources=None):
    """Starts a Slurm job that runs the specified shell command line.
    """

    job_resources_lines = []
    if job_resources is not None:
        for resource, value in job_resources.items():
            job_resources_lines += ["#SBATCH --{}={}".format(resource, value)]

    script_lines = [
            "#!/bin/sh",
            "#SBATCH --output={}".format(outpath)
        ] + job_resources_lines + [
            *additional_setup_lines,
            "srun {}".format(cmdline),
        ]
    return submit_text('\n'.join(script_lines))
