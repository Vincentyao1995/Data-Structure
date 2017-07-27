#!python2.7
""" DOCSTING HERE """

from __future__ import division, print_function

import os
from os.path import exists
import ConfigParser as configparser
import shutil

__author__ = "Matthew Dirks"
__email__ = "mdirks@minesense.com; matt@skylogic.ca"

def getConfigOrCopyDefaultFile(configFpath, defaultFpath):
	cfgDict, cfg = None, None
	err = ''
	if (exists(configFpath)):
		cfgDict, cfg, err = getConfig(configFpath)
	elif (defaultFpath is not None):
		if (exists(defaultFpath)):
			# config file doesn't exist, so we will copy a default configuration file in order to create it
			shutil.copyfile(defaultFpath, configFpath)
			return getConfigOrCopyDefaultFile(configFpath, defaultFpath=None) # read the copied file
		else:
			err = 'Config file (%s) does not exist, and base (default) config file also does not exist (%s).' % (str(configFpath), str(defaultFpath))
	else:
		# config file doesn't exist, and we are not creating a new one by copying from somewhere else
		err = 'Config file (%s) does not exist.' % str(configFpath)

	return cfgDict, cfg, err


def getConfig(configFpath):
	if (not exists(configFpath)):
		errorMessage = 'config file doesn\'t exist'
		return None, None, errorMessage
	else:
		# read existing config file
		cfg = configparser.SafeConfigParser()
		cfg.read(configFpath)

		data = {}
		for sectionName in cfg.sections():
			data[sectionName] = dict(cfg.items(sectionName))

			# for optionName in cfg.options(sectionName):
			# 	val = cfg.get(sectionName, optionName)
			# 	data[sectionName][optionName] = val

		return data, cfg, ''