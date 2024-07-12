import sys
from datetime import datetime


class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1)  # Enable line buffering for the log file
        self.log.write("\n\n")  # Ensure new logs are separated if appending

    def write(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if message != '\n':
            message = f'[{timestamp}] {message}'
        self.terminal.write(message)
        self.log.write(message)
        self.flush()  # Explicitly flush after each write

    def flush(self):
        self.terminal.flush()
        self.log.flush()

