import cloudpickle
from .util import warn_after

WARNING_TIMEOUT = 10 * 60 # seconds

@warn_after("cloudpickle.dumps", WARNING_TIMEOUT)
def dumps(*args, **kwargs):
	return cloudpickle.dumps(*args, **kwargs)

@warn_after("cloudpickle.loads", WARNING_TIMEOUT)
def loads(*args, **kwargs):
	return cloudpickle.loads(*args, **kwargs)