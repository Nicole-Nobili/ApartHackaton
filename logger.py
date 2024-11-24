import logging
import threading

# logging.basicConfig(filename='scorer_logs.txt', level=logging.INFO, format='%(message)s')

# Create a thread-safe logging handler
class ConcurrentLogHandler(logging.Handler):
    def __init__(self, filename):
        super().__init__()
        self._handler = logging.FileHandler(filename)
        self._handler.setFormatter(logging.Formatter('%(message)s'))
        self._lock = threading.Lock()

    def emit(self, record):
        with self._lock:
            self._handler.emit(record)

# Configure concurrent logging for main logs
concurrent_log_handler = ConcurrentLogHandler('scorer_logs.txt')
logger = logging.getLogger('scorer_logger')
logger.setLevel(logging.DEBUG)
logger.addHandler(concurrent_log_handler)