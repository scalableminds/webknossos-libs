import os
if 'USE_CLOUDPICKLE' in os.environ:
	import cloudpickle
	pickle_strategy = cloudpickle
else:
	import pickle
	pickle_strategy = pickle

from .util import warn_after

WARNING_TIMEOUT = 10 * 60 # seconds

@warn_after("pickle.dumps", WARNING_TIMEOUT)
def dumps(*args, **kwargs):
	return pickle_strategy.dumps(*args, **kwargs)

@warn_after("pickle.loads", WARNING_TIMEOUT)
def loads(*args, **kwargs):
	return pickle_strategy.loads(*args, **kwargs)
