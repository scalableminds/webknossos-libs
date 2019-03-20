"""Tools for executing remote commands."""
import cloudpickle
import sys
import os
import traceback
from .util import local_filename

INFILE_FMT = local_filename('cfut.in.%s.pickle')
OUTFILE_FMT = local_filename('cfut.out.%s.pickle')

def format_remote_exc():
    typ, value, tb = sys.exc_info()
    tb = tb.tb_next  # Remove root call to worker().
    return ''.join(traceback.format_exception(typ, value, tb))

def worker(workerid):
    """Called to execute a job on a remote host."""
    print("worker")
    try:
        with open(INFILE_FMT % workerid, 'rb') as f:
            indata = f.read()
        fun, args, kwargs = cloudpickle.loads(indata)
        result = True, fun(*args, **kwargs)
        out = cloudpickle.dumps(result, True)

    except Exception as e:
        print(traceback.format_exc())

        result = False, format_remote_exc()
        out = cloudpickle.dumps(result, False)

    destfile = OUTFILE_FMT % workerid
    tempfile = destfile + '.tmp'
    with open(tempfile, 'wb') as f:
        f.write(out)
    os.rename(tempfile, destfile)

if __name__ == '__main__':
    worker(*sys.argv[1:])
