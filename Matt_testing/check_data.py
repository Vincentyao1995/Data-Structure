#!/bin/env/python3
# http://pythonhosted.org/spectrum/

import spectral as sp
import matplotlib.pyplot as plt
from os.path import join, split, realpath
import pandas as pd

from dvm.config_ini import getConfigOrCopyDefaultFile
from dvm import load_REE_SWIR

######## LOAD CONFIG ########
APP_DIR = join(split(realpath(__file__))[0], '.')
configData, config, err = getConfigOrCopyDefaultFile(join(APP_DIR, 'config.ini'), join(APP_DIR, 'config-default.ini'))

if (configData is None):
	print('ERROR loading config. ' + str(err))

owncloud = configData['data']['owncloud_dir']
data_and_research = join(owncloud, 'data-and-research')

######## SWIR DATA ########
print('### SWIR data')

img = load_REE_SWIR.load_SWIR_image(configData)
mineralogy_df, spectrallyActiveMinerals, targetMinerals = load_REE_SWIR.load_mineralogy(configData)

print(img)
input('Press ENTER to continue...')

# testing
fig, axs = plt.subplots(1,2,figsize=(15,15))
m = img[:,:,100].squeeze()
axs[0].imshow(m)
axs[0].set_title('SWIR image, band 100')
m = img[:,:,200].squeeze()
axs[1].imshow(m)
axs[1].set_title('SWIR image, band 200')
plt.show()

print('Mineralogy according to Rietveld on XRD')
print(mineralogy_df)

print('Spectrally active minerals: ' + str(spectrallyActiveMinerals))
print('Target minerals: ' + str(targetMinerals))
input('Press ENTER to continue...')

print('### Masked SWIR data')
rocks_df, spectrum_columns = load_REE_SWIR.load_SWIR_masked(configData)
for (group_box_number, group_sample_number), group in rocks_df.groupby(['box_number', 'sample_number']):
	print('ROCK: box %s, sample %s, number of spectra %d' % (group_box_number, group_sample_number, group.shape[0]))
input('Press ENTER to continue...')

######## USGS MINERALS ########
print('### USGS minerals')
# USGS pure-ish library of minerals
usgsLib_fpath = join(data_and_research, 'USGSSpectralLibrary/USGS_Resampled_to_nm.hdr')

usgsLib = sp.open_image(usgsLib_fpath)
# usgsLib is an instance of spectral.io.envi.SpectralLibrary

print(usgsLib.names[:10])

print('# names: ', len(usgsLib.names))
print('spectra shape: ', usgsLib.spectra.shape)
print('Note: this spectrum has resolution of 420')

eg = 'kaolini1.spc Kaolinite CM9'
idx = usgsLib.names.index(eg)
kaoloniteSpectrum = usgsLib.spectra[idx, :]

plt.plot(kaoloniteSpectrum); plt.show()

######## REE MINERAL LIBRARY ########
print('### REE mineral library')
# These are pure-ish REE minerals (12 of them) and their spectra
# Spectral resolution: 2151
orderedREE_fpath = join(data_and_research, 'SpectraForAbsorptionFitting_example/OrderedREESpectra_Corr.hdr')
orderedREE = sp.open_image(orderedREE_fpath)

print(orderedREE.names)

for row in orderedREE.spectra:
	plt.plot(row)
plt.show()

######## EXAMPLE SPECTRA ########
print('### Example spectra')
# Some example spectra (just 4 samples, resolution 784)
eg_fpath = join(data_and_research, 'SpectraForAbsorptionFitting_example/SpectraForAbsorptionFitting.hdr')
eg = sp.open_image(eg_fpath)

for row, name in zip(eg.spectra, eg.names):
	plt.plot(row, label=name)
plt.legend()
plt.show()