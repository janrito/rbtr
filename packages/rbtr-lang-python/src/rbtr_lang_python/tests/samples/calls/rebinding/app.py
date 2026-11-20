try:
    from fast import loads
except ImportError:
    from slow import loads

from one import first as pick
from two import second as pick


def run(text):
    return loads(text), pick()
