from functools import wraps
import time
import traceback


def raise_exception(f):
    # type: (object) -> object
    @wraps(f)
    def wrapped(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print traceback.format_exc()
            raise Exception("got %s, from %s with args: %s" % (
                e.__class__.__name__, f.__name__, args)
                            )

    return wrapped


def print_exception(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print("%s, from %s with message: %s with args : %s" % (
                e.__class__.__name__, f.__name__, str(e),args)
                  )
            print traceback.format_exc()

    return wrapped


def timeit(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        t_start = time.time()
        result = f(*args, **kwargs)
        t_end = time.time()
        print 'Time taken by %r with args (%r, %r) : %2.2f sec' % (f.__name__, args, kwargs, t_end - t_start)
        return result

    return wrapped
