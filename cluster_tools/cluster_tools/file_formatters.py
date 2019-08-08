from .util import local_filename
from os import path

def format_infile_name(cfut_dir, job_id):
	return path.join(cfut_dir, "cfut.in.%s.pickle" % job_id)

def format_outfile_name(cfut_dir, job_id):
	return path.join(cfut_dir, "cfut.out.%s.pickle" % job_id)
