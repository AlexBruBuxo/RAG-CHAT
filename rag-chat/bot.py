import logging
import sys

LOGGING_LEVEL = logging.DEBUG

logging.basicConfig(stream=sys.stdout, level=LOGGING_LEVEL)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


