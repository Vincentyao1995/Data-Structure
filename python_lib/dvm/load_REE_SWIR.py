import spectral as sp
from os.path import join, split, realpath
import pandas as pd
from io import StringIO
import re
import numpy as np
import os
import gzip
import pickle

from arca.plot.colorPalette import getNColorsAsHex

def load_SWIR_image(configData):

	# location of contents of SWIR.zip (downloaded from OwnCloud)
	swir_data_dir = configData['data']['swir_data_dir']

	SWIR_fpath = join(swir_data_dir, 'L12464-Box120_SawnQuantSuite_SWIR_High.hdr')
	roi1_fpath = join(swir_data_dir, 'L12464-Box120_SawnQuantSuite_SWIR_High_SampleROI.roi')
	roi2_fpath = join(swir_data_dir, 'L12464-Box120_SawnQuantSuite_SWIR_High_MergedAllSampleOutlines_ROIs.roi')

	print('Samples are numbered 1 to 6 from top to bottom.')
	img = sp.open_image(SWIR_fpath)

	return img

def load_mineralogy(configData):
	owncloud = configData['data']['owncloud_dir']
	data_and_research = join(owncloud, 'data-and-research')

	# mineralogy according to Rietveld on XRD:
	REEdata_fpath = join(data_and_research, 'REEBearingRocks_ImageCube_Data/Quantitative REE Approximate Worksheets.xlsx')
	REEdata_df = pd.read_excel(REEdata_fpath, 'Working', header=1)

	sixSamples = ['L12_464 Box120-1','L12_464 Box120-2','L12_464 Box120-3','L12_464 Box120-4','L12_464 Box120-5','L12_464 Box120-6']
	mineralogy_df = REEdata_df[sixSamples].iloc[:18]

	# remove aspectral minerals
	aspectralMinerals = ['Albite low', 'Microcline (ordered)', 'Quartz low', 'Aegirine/Augite?']
	spectrallyActiveMinerals = set(mineralogy_df.index) - set(aspectralMinerals)

	# remove minerals with all 0's (no content in all the rocks)
	mineralsWithAllZeros = set()
	for pdIdx, row in mineralogy_df.iterrows():
	    if (all(row==0)):
	        mineralsWithAllZeros.add(pdIdx)
	targetMinerals = spectrallyActiveMinerals - mineralsWithAllZeros

	return mineralogy_df, spectrallyActiveMinerals, targetMinerals

def load_SWIR_masked(configData):
	""" Load image data that has been masked in ENVI and exported as ASCII.
	Example first 10 lines of data file:
		; ENVI Output of ROIs (5.0) [Fri Jul 21 10:23:58 2017]
		; Number of ROIs: 1
		; File Dimension: 320 x 3732
		;
		; ROI name: Box120 Sample 1 SWIR
		; ROI rgb value: {255, 0, 0}
		; ROI npts: 60073
		      ID    X    Y      B1      B2      B3      B4      B5      B6      B7      B8      B9     B10  and so on
		       1  290   68  0.0642  0.1180  0.1337  0.1558  0.1588  0.1592  0.1625  0.1639  0.1622  0.1646  ...
		       2  291   68  0.0998  0.1340  0.1441  0.1690  0.1679  0.1744  0.1845  0.1846  0.1817  0.1843  ...
	"""
	swir_masked_dir = configData['data']['swir_masked_dir']
	image_fnames = ['Box120_SWIR_sample5.txt','Box120_SWIR_sample6.txt','Box120_SWIR_sample1.txt','Box120_SWIR_sample2.txt','Box120_SWIR_sample3.txt','Box120_SWIR_sample4.txt',]
	fname_pat = re.compile('Box(?P<box_number>\d+)_SWIR_sample(?P<sample_number>\d+)\.txt')

	# some arbitrary colors
	colors = getNColorsAsHex(len(image_fnames))

	rock_dfs = []
	for fname, color in zip(image_fnames, colors):
		rock_data = fname_pat.match(fname).groupdict()

		masked_fpath = join(swir_masked_dir, fname)

		# first, remove the leading semi-colon (;) from the header row (line 8) in the file.
		# this makes it conform to the format that Pandas is expecting.
		f = open(masked_fpath, 'r')
		txt = f.read().replace(';', ' ', 7)
		buf = StringIO(txt)

		# Header is expected to be in line 8 (7 if 0-indexed). Preceding lines contain metadata generated by ENVI.
		# Skip the first 7 lines.
		for i in range(7): buf.readline()

		# load into pandas
		# Columns are fixed-format (delim_whitespace=True)
		# Note: setting header=7 didn't seem to work with a buffer (but did when read directly from file)
		masked_df = pd.read_table(buf, delim_whitespace=True)
		masked_df['box_number'] = rock_data['box_number']
		masked_df['sample_number'] = rock_data['sample_number']
		masked_df['color'] = color

		# save
		rock_dfs.append(masked_df)
		print('Done loading ' + fname)

	df = pd.concat(rock_dfs)
	df.index = np.arange(len(df))
	spectrum_columns = ['B%d' % i for i in range(1, 256+1)]

	return df, spectrum_columns

def load_from_cache(configData, CACHE_FPATH):
	""" Call load_SWIR_masked() and load_mineralogy(),
	and cache the result.
	If the cache exists, it will be used instead of loading everything again.
	"""

	if (os.path.exists(CACHE_FPATH)):
		with gzip.open(CACHE_FPATH, 'rb') as f:
			cachedData = pickle.load(f)
		print('Rocks loaded from cache')
	else:
		rocks_df, spectrumColumns = load_SWIR_masked(configData)
		print('Rocks loaded from files')

		mineralogy_df, spectrallyActiveMinerals, targetMinerals = load_mineralogy(configData)

		cachedData = {
			'rocks_df': rocks_df,
			'spectrumColumns': spectrumColumns,
			'mineralogy_df': mineralogy_df,
			'targetMinerals': targetMinerals,
			'spectrallyActiveMinerals': spectrallyActiveMinerals,
		}
		with gzip.open(CACHE_FPATH, 'wb') as f:
			pickle.dump(cachedData, f)

		print('Rocks saved to cache ' + CACHE_FPATH)

	return cachedData