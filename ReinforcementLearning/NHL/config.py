# -*- coding: utf-8 -*-
"""Handles per-user configuration.

Example:

Attributes:

Todo:
    * Nothing for now.

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import configparser
from pathlib import Path
from Utils.base import append_to_git_root
import os
class Config(object):

    def __init__(self):
        self.file_name=append_to_git_root(what="ReinforcementLearning/NHL/config.ini", alternate_root="/tmp")
        assert Path(self.file_name).is_file(), "Config file '%s' does not exist." % self.file_name
        # Load the configuration file

        config = configparser.ConfigParser()
        config.read(self.file_name)
        self.data_dir=config['data']['folder']
        assert os.path.isdir(self.data_dir), "'%s' does not exist or is not a directory." % (self.data_dir)
