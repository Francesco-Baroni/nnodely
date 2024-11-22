import importlib.metadata
__version__ = importlib.metadata.version(__package__)

import sys
major, minor = sys.version_info.major, sys.version_info.minor

import logging
LOG_LEVEL = logging.INFO

if major < 3:
    sys.exit("Sorry, Python 2 is not supported. You need Python >= 3.6 for "+__package__+".")
elif minor < 10:
    sys.exit("Sorry, You need Python >= 3.10 for "+__package__+".")
else:
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>---- {__package__}_v{__version__} ----<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')