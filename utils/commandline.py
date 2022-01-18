import argparse
from importlib.machinery import SourceFileLoader


def load_config(argv):

    config = SourceFileLoader('config', "./config/config.py").load_module()
    return config


def load_config_test(argv):

    config_test = SourceFileLoader('config_test', "./config/config_test.py").load_module()
    return config_test