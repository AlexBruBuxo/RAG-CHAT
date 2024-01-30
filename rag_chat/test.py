import logging

# Note: This must be set at the start of each file when we want 
# to log something (this will show the name of the logger in the log)
logger = logging.getLogger(__name__)

def test():
    logger.info("Test info message")