import logging
import sys

from srunner.osc2.osc_preprocess.import_msg import create_ImportMsg as import_msg

try:
    from tests.run_testcase.log_msg import create_LogMsg as log_msg
except ImportError:
    log_msg = None

ERROR_WITH_EXIT = False
ERROR_COUNT = 0
ERROR_MAX_COUNT = 10  # Output the maximum number of errors
LOG_LEVEL = logging.ERROR  # The lowest level of output logs
LOG_FORMAT = "%(message)s "  # Output log format
DATE_FORMAT = "%Y-%m-%d  %H:%M:%S %a "  # Format of the output time

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
)


def LOG_DEBUG(msg, token=None, line=None, column=None):
    if token is not None:
        file_path, line = import_msg.get_msg(token.line)
        msg = (
            '[Debug] file "'
            + file_path
            + '", line '
            + str(line)
            + ":"
            + str(token.column)
            + ", "
            + msg
        )
    elif line is not None and column is not None:
        file_path, line = import_msg.get_msg(line)
        msg = (
            '[Debug] file "'
            + file_path
            + '", line '
            + str(line)
            + ":"
            + str(column)
            + ", "
            + msg
        )
    logging.debug(msg)


def LOG_INFO(msg, token=None, line=None, column=None):
    if token is not None:
        file_path, line = import_msg.get_msg(token.line)
        msg = (
            '[Info] file "'
            + file_path
            + '", line '
            + str(line)
            + ":"
            + str(token.column)
            + ", "
            + msg
        )
    elif line is not None and column is not None:
        file_path, line = import_msg.get_msg(line)
        msg = (
            '[Info] file "'
            + file_path
            + '", line '
            + str(line)
            + ":"
            + str(column)
            + ", "
            + msg
        )
    logging.info(msg)


def LOG_WARNING(msg, token=None, line=None, column=None):
    # Log information required when running run_symbol_testcases.py
    run_log_msg = ""
    if token is not None:
        file_path, line = import_msg.get_msg(token.line)
        run_log_msg = (
            "[Warning]" + " line " + str(line) + ":" + str(token.column) + ", " + msg
        )
        msg = (
            '[Warning] file "'
            + file_path
            + '", line '
            + str(line)
            + ":"
            + str(token.column)
            + ", "
            + msg
        )
    elif line is not None and column is not None:
        file_path, line = import_msg.get_msg(line)
        run_log_msg = (
            "[Warning]" + " line " + str(line) + ":" + str(column) + ", " + msg
        )
        msg = (
            '[Warning] file "'
            + file_path
            + '", line '
            + str(line)
            + ":"
            + str(column)
            + ", "
            + msg
        )
    if log_msg and log_msg.is_open:
        log_msg.add_log_msg(run_log_msg)
    else:
        logging.warning(msg)


def LOG_ERROR(msg, token=None, line=None, column=None):
    global ERROR_WITH_EXIT
    global ERROR_COUNT
    global ERROR_MAX_COUNT

    # Log information required when running run_symbol_testcases.py
    run_log_msg = ""
    if token is not None:
        file_path, line = import_msg.get_msg(token.line)
        run_log_msg = (
            "[Error]" + " line " + str(line) + ":" + str(token.column) + ", " + msg
        )
        msg = (
            '[Error] file "'
            + file_path
            + '", line '
            + str(line)
            + ":"
            + str(token.column)
            + ", "
            + msg
        )
    elif line is not None and column is not None:
        file_path, line = import_msg.get_msg(line)
        run_log_msg = "[Error]" + " line " + str(line) + ":" + str(column) + ", " + msg
        msg = (
            '[Error] file "'
            + file_path
            + '", line '
            + str(line)
            + ":"
            + str(column)
            + ", "
            + msg
        )
    # If you run run_symbol_testcases.py, no error is reported
    if log_msg and log_msg.is_open:
        log_msg.add_log_msg(run_log_msg)
    else:
        logging.error(msg)
        ERROR_COUNT += 1
        if ERROR_COUNT >= ERROR_MAX_COUNT:
            if ERROR_WITH_EXIT:
                sys.exit(1)
            else:
                pass
        else:
            pass


def LOG_CRITICAL(msg, token=None, line=None, column=None):
    global ERROR_WITH_EXIT
    global ERROR_COUNT
    global ERROR_MAX_COUNT

    if token is not None:
        file_path, line = import_msg.get_msg(token.line)
        msg = (
            '[Error] file "'
            + file_path
            + '", line '
            + str(line)
            + ":"
            + str(token.column)
            + ", "
            + msg
        )
    elif line is not None and column is not None:
        file_path, line = import_msg.get_msg(line)
        msg = (
            '[Error] file "'
            + file_path
            + '", line '
            + str(line)
            + ":"
            + str(column)
            + ", "
            + msg
        )
    logging.critical(msg)
    ERROR_COUNT += 1
    if ERROR_COUNT >= ERROR_MAX_COUNT:
        if ERROR_WITH_EXIT:
            sys.exit(1)
        else:
            pass
    else:
        pass
