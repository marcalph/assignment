#!/usr/bin/env python3
# coding: utf-8
##########################################
# authors                                #
# marcalph - https://github.com/marcalph #
##########################################
""" log utilities
"""

import functools
import logging
import os
import time
from inspect import getframeinfo, stack


class CustomFormatter(logging.Formatter):
    """Custom formatter, overrides funcname and filename if provided"""

    def format(self, record):
        if hasattr(record, "funcname_override"):
            record.funcname = record.funcname_override
        return super(CustomFormatter, self).format(record)


def logthis(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        py_file_caller = getframeinfo(stack()[1][0])
        extra_args = {
            "funcname_override": os.path.basename(py_file_caller.filename)
            + "/"
            + fn.__name__
        }
        logger.info("started", extra=extra_args)
        t = time.time()
        function = fn(*args, **kwargs)
        logger.info(f"ended in {time.time()-t:.1f} sec", extra=extra_args)
        return function

    return wrapper


logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
# fh = logging.FileHandler("assets/output/logs/models.log")
formatter = CustomFormatter(
    "%(asctime)s - %(levelname)s - %(funcname)s - %(message)s", "%Y-%m-%d"
)
ch.setFormatter(formatter)
# fh.setFormatter(formatter)
logger.addHandler(ch)
# logger.addHandler(fh)
