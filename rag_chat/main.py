import logging
from rag_chat.log import setup_logging

# Note: This should be added at the start of the app to control the logging.
setup_logging()

# Note: This must be set at the start of each file when we want 
# to log something (this will show the name of the logger in the log)
logger = logging.getLogger(__name__)


