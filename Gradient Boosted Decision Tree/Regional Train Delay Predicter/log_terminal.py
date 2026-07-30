
#* terminal_logger.py

#* ======= Library Imports =======
# ------- Native Libraries. -------
import sys
from typing import TextIO

# ------- External Library. -------
from pathlib import Path

class TerminalLogger:
    """
    A utility class to duplicate stream output to both terminal and disk.

    This class acts as a dual-destination stream wrapper. Any text sent to it
    (typically from print statements or uncaught exception tracebacks) is written 
    both to the standard terminal screen and appended to a log file on disk .

    Attributes:
        terminal (TextIO): A reference to the original system standard output stream .
        log_file (TextIO): The open text file stream used to save log outputs .
    """
    def __init__(
            self,
            file_path: Path
    ) -> None:
        """
        Initializes the logger and opens the target text file .

        Args:
            file_path (Path): The filesystem path where the log file should be 
                created or appended .
        """
        # Keeps a reference to the original terminal screen output.
        self.terminal: TextIO = sys.stdout

        # Opens the log file in append 'a' mode.
        self.log_file: TextIO = open(
            file = file_path,
            mode = "a",
            encoding = "utf-8"
        )

    def write(
            self,
            message: str
    ) -> None:
        """
        Writes the text message to both stdout and the target log file .

        Args:
            message (str): The string content to be written .
        """
        # Writes to the terminal screen.
        self.terminal.write(message)
        # Simultaneously writes to the .txt file.
        self.log_file.write(message)

        self.flush()

    def flush(self) -> None:
        """
        Flushes the stream buffers for both terminal and disk file .
        asdas

        This ensures that any text currently residing in the system buffers is
        immediately printed to the screen and committed to the file on disk .
        """
        # Flushes buffers to ensure text is immediately written.
        self.terminal.flush()
        self.log_file.flush()

    def close(self) -> None:
        """
        Closes the underlying log file stream .

        This should be called at the end of the script execution to release the
        file resources back to the operating system .
        """
        # Closes the .text file stream when done.
        self.log_file.close()