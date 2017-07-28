# http://pythonhosted.org/spectrum/

import spectral as sp
import matplotlib.pyplot as plt
from os.path import join, split, realpath
import pandas as pd

from dvm.config_ini import getConfigOrCopyDefaultFile

######## LOAD CONFIG ########
APP_DIR = join(split(realpath(__file__))[0], '.')
configData, config, err = getConfigOrCopyDefaultFile(join(APP_DIR, 'config.ini'), join(APP_DIR, 'config-default.ini'))

if (configData is None):
	print('ERROR loading config. ' + str(err))

owncloud = configData['data']['owncloud_dir']
data_and_research = join(owncloud, 'data-and-research')

######## SWIR DATA ########
print('### SWIR data')

# location of contents of SWIR.zip (downloaded from OwnCloud)
swir_data_dir = configData['data']['swir_data_dir']

SWIR_fpath = join(swir_data_dir, 'L12464-Box120_SawnQuantSuite_SWIR_High.hdr')
roi1_fpath = join(swir_data_dir, 'L12464-Box120_SawnQuantSuite_SWIR_High_SampleROI.roi')
roi2_fpath = join(swir_data_dir, 'L12464-Box120_SawnQuantSuite_SWIR_High_MergedAllSampleOutlines_ROIs.roi')

print('Samples are numbered 1 to 6 from top to bottom.')
img = sp.open_image(SWIR_fpath)
print(img)

# testing
m = img[:,:,100].squeeze()
fig, ax = plt.subplots(1,1,figsize=(15,15))
ax.imshow(m)
plt.show()
# testing
m = img[:,:,200].squeeze()
fig, ax = plt.subplots(1,1,figsize=(15,15))
ax.imshow(m)
plt.show()

# Corresponding mineralogy:
REEdata_fpath = join(data_and_research, 'REEBearingRocks_ImageCube_Data/Quantitative REE Approximate Worksheets.xlsx')
REEdata_df = pd.read_excel(REEdata_fpath, 'Working', header=1)

sixSamples = ['L12_464 Box120-1','L12_464 Box120-2','L12_464 Box120-3','L12_464 Box120-4','L12_464 Box120-5','L12_464 Box120-6']
mineralogy_df = REEdata_df[sixSamples].iloc[:18]

print(mineralogy_df)

aspectralMinerals = ['Albite low', 'Microcline (ordered)', 'Quartz low', 'Aegirine/Augite?']
spectrallyActiveMinerals = set(mineralogy_df.index) - set(aspectralMinerals)

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