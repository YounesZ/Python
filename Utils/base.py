#!/usr/bin/env python
"""Several Utilities.

"""

import logging
from logging.handlers import RotatingFileHandler

class hashable_dict(dict):
    """A dictionary that is hashable."""

    def __hash__(self):
        return hash(tuple(sorted(self.items())))

def get_logger(name: str, debug_log_file_name: str): # -> logging.Logger:
    """
    Returns or creates Logger identified to a certain name.
    :param name: the identifier for the logger.
    :param debug_log_file_name: Where we will write debugging information.
    :return: a logger that can be used in the 'normal' way (ie, logger.info("hi, there")).
    """
    alogger = logging.getLogger(name)
    alogger.setLevel(logging.DEBUG) # CAREFUL ==> need this, otherwise everybody chokes!
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s [%(module)s.%(funcName)s:%(lineno)d => %(message)s]')
    #
    create_debug_handler = False
    # fh = logging.FileHandler(debug_log_file_name)
    fh = RotatingFileHandler(debug_log_file_name, mode='a', maxBytes=5 * 1024 * 1024, backupCount=2, encoding=None, delay=0)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    if not len(alogger.handlers):
        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        # and add it to logger
        alogger.addHandler(ch)

        # we need a debug handler: let's flag our needs:
        create_debug_handler = True

        print("Logger created")
    else:
        print("Logger retrieved")
        # did the file handler change names?
        curr_debug_handler = alogger.handlers[1]
        if curr_debug_handler.baseFilename != fh.baseFilename:
            print("Changing log file names; was '{}', switching to '{}'".format(curr_debug_handler.baseFilename,
                                                                                fh.baseFilename))
            alogger.removeHandler(curr_debug_handler)
            # we need a debug handler: let's flag our needs:
            create_debug_handler = True
        else:
            # the debug handler we have is all good!
            create_debug_handler = False

    # If we need a debug handler, let's create it!
    if create_debug_handler:
        print("Creating debug handler at '{}'".format(fh.baseFilename))
        alogger.addHandler(fh)

    s = "'{}': logging 'INFO'+ logs to Console, 'DEBUG'+ logs to '{}'".format(alogger.name, alogger.handlers[1].baseFilename)
    print(s)
    alogger.info(s)
    alogger.debug(s)
    return alogger

