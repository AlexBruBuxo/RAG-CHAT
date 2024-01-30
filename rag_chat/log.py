import logging
import sys
import os
from dotenv import load_dotenv

load_dotenv()

def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)

    logging.basicConfig(stream=sys.stdout, level=level,
                        format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
