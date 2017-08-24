#!/usr/bin/env python3
""" Iteratively create clusters, using arbitrary threshold (more memory-friendly than spectral clustering) """

from __future__ import division, print_function
import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue, Pipe, Pool
import time
from os.path import join, split, realpath, dirname
import logging
from collections import OrderedDict
from colorama import Fore, Back, Style
import colorama
from os.path import join, split, realpath
import os
from tqdm import tqdm
import pickle
from itertools import repeat
import matplotlib.ticker as plticker
import psutil
import sys

from arca.plot.subplotShape import makeNiceGridFromNElements
from arca.plot.colorPalette import getNColorsAsHex
from arca.vectorAngle import angle
from mtl.file_io.make_output_dir import newDir
from mtl.file_io.archive_source_files import zipSources
from dvm import load_REE_spectra
from dvm.config_ini import getConfig
from arca.spectral_clustering.spectral_clustering_helpers import makeClustersHistogramFigure

__author__ = "Matthew Dirks"
__email__ = "mdirks@minesense.com; matt@skylogic.ca"

RESULTS_DIR = 'results/clustering_iterative'
CACHE_FPATH = 'cache/hyperspectral_NN_rocks.pkl.gz'
configFpath = "config.ini"

# ANGLE_THRESHOLD = 0.04 # had 200 clusters by 70,000 (too many)
ANGLE_THRESHOLD = 0.1

loc = plticker.MultipleLocator(base=1.0)

def makePlots(wavelengths, cluster_members_lists, cluster_exemplars, rocks_df, spectrumColumns):
	print('Plot members\'s spectra')

	nClusters= 0
	clusterIndicesToPlot = []
	for idx, exemplar in enumerate(cluster_exemplars):
		if (exemplar is not None):
			nClusters += 1
			clusterIndicesToPlot.append(idx)

	colors = getNColorsAsHex(20); nc = len(colors)

	(nRows, nCols), _ = makeNiceGridFromNElements(nClusters)
	fig, axs = plt.subplots(nRows, nCols, figsize=(15,10))

	if (isinstance(axs, mpl.axes._subplots.Subplot)):
		axs = np.array([axs])
	elif (len(axs.shape) == 2): # 2D array, needs to be flattened
		axs = [item for sublist in axs for item in sublist]

	clusterAverages = []
	for clusterIdx, ax in zip(clusterIndicesToPlot, axs):
		members = cluster_members_lists[clusterIdx]
		exemplar = cluster_exemplars[clusterIdx]

		# plot members's spectra
		spectra = rocks_df.loc[members][spectrumColumns]
		# only some
		spectra = spectra.sample(n=min(400, len(spectra)))
		for pdIdx, s in spectra.iterrows():
			ax.plot(wavelengths, s.tolist(), c=colors[pdIdx%nc], alpha=0.05)

		# plot cluster exemplar
		ax.plot(wavelengths, exemplar.tolist(), c='k', label='mean')

		ax.set_title('clusterIdx=%d' % clusterIdx)

	fig.tight_layout()
	fig.savefig(join(RESULTS_DIR, 'clusters_%d_spectra.png' % nClusters))
	plt.close(fig)

	# fig = makeClustersHistogramFigure(rocks_df, 'predictedCluster_kmeans', instanceIdColumn='sample_number')
	if ('predictedCluster_iterative' in rocks_df):
		print('Plot clusters histogram')
		fig = makeClustersHistogramFigure(rocks_df, 'predictedCluster_iterative')
		fig.axes[0].xaxis.set_major_locator(loc)
		fig.savefig(join(RESULTS_DIR, 'clusters_%d_hist.png' % nClusters))
		plt.close(fig)


def load():
	t0 = time.time()
	cachedData = load_REE_SWIR.load_from_cache(configData, CACHE_FPATH)
	print('time %0.3f s (to load rocks from cache)' % (time.time() - t0))
	spectrumColumns = cachedData['spectrumColumns']
	rocks_df = cachedData['rocks_df']

	# rocks_df = rocks_df.sample(frac=0.017) # ~10,000, 2hours
	# rocks_df = rocks_df.sample(frac=0.034)
	print('num samples selected: ', len(rocks_df))

	# last column of image has NaN often, lets just chop it off
	# spectrumColumns = spectrumColumns[:-1]

	# DT recommends excluding first 4 or 5 bins, as they are often extremely high/low (maybe due to efficiency of the sensor at the extremes?)
	# The last few probably aren't very good either -Matt.
	spectrumColumns = spectrumColumns[5:-5]

	print('loading wavelengths...')
	wavelengths = load_REE_SWIR.load_SWIR_wavelengths(configData)
	wavelengths = wavelengths[5:-5]

	return rocks_df, spectrumColumns, wavelengths

def get_angle(exemplar, spectrum):
	return angle(spectrum, exemplar)

def check_memory():
	usedGB = psutil.virtual_memory().used/1024.**3
	percent = psutil.virtual_memory().percent

	if (usedGB > 7 or percent > 50):
		if (input('WARNING: high RAM usage. Proceed? ') in 'Yy'):
			return
		else:
			quit()

def getExistingClusterData():
	with open(input('Full path to clusters.pkl: '), 'rb') as f:
		data = pickle.load(f)
	cluster_members_lists = data['cluster_members_lists']
	cluster_exemplars = data['cluster_exemplars']

	return cluster_members_lists, cluster_exemplars

def plotExistingClusterAssignments(rocks_df, spectrumColumns, wavelengths):
	rocks_df['predictedCluster_iterative'] = -1 # initialize (and imply integer dtype)

	print('loading clusterings...')
	cluster_members_lists, cluster_exemplars = getExistingClusterData()

	print('recording cluster assignments...')
	for clusterIdx, members in enumerate(cluster_members_lists):
		# record cluster assignment to these members
		rocks_df.loc[members, 'predictedCluster_iterative'] = clusterIdx

	makePlots(wavelengths, cluster_members_lists, cluster_exemplars, rocks_df, spectrumColumns)

def getBestClusterMatch(spectrum, cluster_exemplars):
	### Three different ways of computing all the angles
	# (I found the first one to be the fastest)

	def getAngleToSpectrum(exemplar):
		if (exemplar is None):
			return sys.float_info.max
		else:
			return angle(spectrum, exemplar)

	if (True):
		anglesToClusters = np.array(list(map(getAngleToSpectrum, cluster_exemplars)))
		bestClusterIdx = anglesToClusters.argmin()
		bestAngle = anglesToClusters[bestClusterIdx]
	if (False):
		bestClusterIdx = None
		bestAngle = 999.9
		for cIdx, exemplar in enumerate():
			a = angle(spectrum, exemplar)
			if (a < bestAngle):
				bestAngle = a
				bestClusterIdx = cIdx
	if (False):
		anglesToClusters = np.array(pool.starmap(get_angle, zip(cluster_exemplars, repeat(spectrum))))
		bestClusterIdx = anglesToClusters.argmin()
		bestAngle = anglesToClusters[bestClusterIdx]

	return bestAngle, bestClusterIdx

def runClustering(rocks_df, spectrumColumns):
	cluster_members_lists = [
		[rocks_df.index[0]],
	]
	cluster_exemplars = [
		rocks_df.iloc[0][spectrumColumns].as_matrix(),
	]

	t1 = time.time()
	for idx, (pdIdx, row) in tqdm(enumerate(rocks_df.iloc[1:].iterrows())):
		spectrum = row[spectrumColumns].as_matrix()

		bestAngle, bestClusterIdx = getBestClusterMatch(spectrum, cluster_exemplars)

		### Given the angles, determine which cluster to add this spectrum to, or start a new cluster
		if (bestAngle < ANGLE_THRESHOLD):
			# add current spectrum to closest cluster
			cluster_members_lists[bestClusterIdx].append(pdIdx)

			# recompute cluster center (exemplar)
			# only sometimes...
			_members = cluster_members_lists[bestClusterIdx]
			if (len(_members) % 100 == 2):
				cluster_exemplars[bestClusterIdx] = rocks_df.loc[_members][spectrumColumns].mean(axis=0)
		else:
			# create new cluster with current spectrum
			cluster_members_lists.append([pdIdx])
			cluster_exemplars.append(spectrum)

			print('[%d/%d] nClusters = %d' % (idx, len(rocks_df), len(cluster_exemplars)))

	# finally, do one last calculation of exemplars
	for clusterIdx, members in enumerate(cluster_members_lists):
		cluster_exemplars[clusterIdx] = rocks_df.loc[members][spectrumColumns].mean(axis=0)

	print('time %0.3f s' % (time.time() - t1))

	toSave = {
		'cluster_members_lists': cluster_members_lists,
		'cluster_exemplars': cluster_exemplars,
	}
	pickle.dump(toSave, open(join(RESULTS_DIR, 'clusters.pkl'), 'wb'))
	print('saved')

def recheck(rocks_df, spectrumColumns):
	cluster_members_lists, cluster_exemplars = getExistingClusterData()

	new_cluster_members_lists = [[] for _ in cluster_members_lists]

	changeCount = 0

	# for each existing cluster
	for clusterIdx, members in enumerate(cluster_members_lists):
		print('rechecking cluster', clusterIdx)

		# for each rock in this cluster, check if there's an even better cluster for it
		for pdIdx in tqdm(members):
			row = rocks_df.loc[pdIdx]
			spectrum = row[spectrumColumns].as_matrix()

			# get existing angle to existing cluster
			currentAngle = angle(spectrum, cluster_exemplars[clusterIdx])

			# get angles to other clusters, and minimize
			bestAngle, bestClusterIdx = getBestClusterMatch(spectrum, cluster_exemplars)

			if (bestAngle < currentAngle):
				changeCount += 1
				new_cluster_members_lists[bestClusterIdx].append(pdIdx)
			else:
				new_cluster_members_lists[clusterIdx].append(pdIdx)

		print('changeCount = ', changeCount)

	print('recheck done')
	print('redoing exemplars...')

	# recompute cluster center (exemplar)
	for clusterIdx, members in enumerate(new_cluster_members_lists):
		cluster_exemplars[clusterIdx] = rocks_df.loc[members][spectrumColumns].mean(axis=0)

	print('preparing to save results...')
	toSave = {
		'cluster_members_lists': new_cluster_members_lists,
		'cluster_exemplars': cluster_exemplars,
	}
	pickle.dump(toSave, open(join(RESULTS_DIR, 'clusters.pkl'), 'wb'))
	print('saved')

def purgeClusters(rocks_df, spectrumColumns):
	toPurge = input('Enter list of cluster indices to be purged (CSV): ')
	# e.g. 8,11,12
	toPurge = [int(x) for x in toPurge.split(',')]

	cluster_members_lists, cluster_exemplars = getExistingClusterData()
	new_cluster_exemplars = cluster_exemplars.copy()
	for i in toPurge:
		new_cluster_exemplars[i] = None # to prevent members from attaching themselves to these purged clusters

	# for each cluster to be purged
	for clusterIdx in toPurge:
		members = cluster_members_lists[clusterIdx]
		print('purging cluster %d with %d members' % (clusterIdx, len(members)))

		# for each rock in this cluster, find closest other cluster
		for pdIdx in tqdm(members):
			row = rocks_df.loc[pdIdx]
			spectrum = row[spectrumColumns].as_matrix()

			# get angles to other clusters, and minimize
			bestAngle, bestClusterIdx = getBestClusterMatch(spectrum, new_cluster_exemplars)
			assert bestClusterIdx not in toPurge
			cluster_members_lists[bestClusterIdx].append(pdIdx)

		# wipe out cluster (that has had all its members moved to another)
		members.clear()

	print('purge done')

	# recompute cluster center (exemplar)
	print('redoing exemplars...')
	for clusterIdx, members in enumerate(cluster_members_lists):
		if (new_cluster_exemplars[clusterIdx] is not None):
			new_cluster_exemplars[clusterIdx] = rocks_df.loc[members][spectrumColumns].mean(axis=0)

	print('preparing to save results...')
	toSave = {
		'cluster_members_lists': cluster_members_lists,
		'cluster_exemplars': new_cluster_exemplars,
	}
	pickle.dump(toSave, open(join(RESULTS_DIR, 'clusters.pkl'), 'wb'))
	print('saved')

def main():
	check_memory()
	
	# pool = Pool(processes=4) 

	rocks_df, spectrumColumns, wavelengths = load()

	if (input('Plot existing cluster assignments?') in list('yY')):
		plotExistingClusterAssignments(rocks_df, spectrumColumns, wavelengths)
	elif (input('Run clustering?') in list('yY')):
		runClustering(rocks_df, spectrumColumns)
	elif (input('Run clustering re-check?') in list('yY')):
		recheck(rocks_df, spectrumColumns)
	elif (input('Run clustering purge?') in list('yY')):
		purgeClusters(rocks_df, spectrumColumns)

if __name__ == '__main__':
	configData, _, _ = getConfig(configFpath)
	main()