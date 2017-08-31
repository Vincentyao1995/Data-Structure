
#this script is do Gaussian modeling for multiple minerals and output the paramters like DT's method. 

import spectral.io.envi as envi
import ModifiedGaussianModel as MGM
import numpy as np
import pre_processing_mineral as ppm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

switch_plotBand = 0

class datasetsPCA():
    def __init__(self):
        pass

# input the filePath and return a spectral.io.envi class including many info.
def open_spectraData(filePath):
    sp_lib = envi.open(filePath)

    return sp_lib

# input the filePath and mineral name u want, return initial Gaussian modeling parameters.
def get_initialParams_fromTXT(filePath, sp_name):
    file = open(filePath, 'r')
    lines = [line for line in file]
    dict_mineral_initParams = {}
    flag_mineralReading = 0
    
    for line in lines:
        if 'Mineral' in line:
            mineralName = line.replace(':','').split(' ')[1]
            dict_mineral_initParams.setdefault(mineralName, {})
            flag_mineralReading = 1
            continue

        if flag_mineralReading == 1:
            if 'band' in line:
                line = line.replace('\n','')
                begin = line.split(' ')[2]
                end = line.split(' ')[-1]
                band = line.split(' ')[0]
                dict_mineral_initParams[mineralName].setdefault(band, {})
                dict_mineral_initParams[mineralName][band].setdefault('begin', float(begin))
                dict_mineral_initParams[mineralName][band].setdefault('end', float(end))
                continue

            elif 'height' in line:
                line = line.replace('\n','')
                height = line.split('\t')[1:]
                height = [float(h) for h in height]
                dict_mineral_initParams[mineralName][band].setdefault('height', height)
                continue

            elif 'width' in line:
                line = line.replace('\n','')
                width = line.split('\t')[1:]
                width = [float(w) for w in width]
                dict_mineral_initParams[mineralName][band].setdefault('width', width)
                continue

            elif 'center' in line:
                line = line.replace('\n','')
                center = line.split('\t')[1:]
                center = [float(c) for c in center]
                dict_mineral_initParams[mineralName][band].setdefault('center', center)
                continue

            elif 'yshift' in line:
                line = line.replace('\n','')
                yshift = line.split('\t')[1]
                dict_mineral_initParams[mineralName][band].setdefault('yshift', float(yshift))
                continue
                
    for name in sorted(dict_mineral_initParams.keys()):
        if name in sp_name:
            return dict_mineral_initParams[name]


def dictWrite(filePath, dict):
    fileOut = open(filePath, 'w')
    
    for mineral in sorted(dict.keys()):
        fileOut.write('Mineral %s\n' % mineral)
        for band in sorted(dict[mineral].keys()):
            fileOut.write('%s Optimal Parameters(Height, Width, Center, Yshift):\n ' % band)
            for i in range(len(dict[mineral][band])):
                numTemp  = int(len(dict[mineral][band])/3)
                if int(i / numTemp)== 0:
                    
                    fileOut.write('%f\t' % dict[mineral][band][i])
                    if int((i+1)/numTemp == 1):
                        fileOut.write('\n')
                
                elif int(i / numTemp) ==1:
                    fileOut.write('%f\t' % dict[mineral][band][i])
                    if int((i+1)/numTemp == 2):
                        fileOut.write('\n')
                elif int(i / numTemp) == 2:
                    fileOut.write('%f\t' % dict[mineral][band][i])
                    if int((i+1)/numTemp == 3):
                        fileOut.write('\n')
                elif int(i / numTemp) == 3:
                    fileOut.write('%f\n' % dict[mineral][band][i])
            
        fileOut.write('\n')
    fileOut.write('')

#this function input a dict including parameter lists from multiple minerals. return the PCA result.
def cal_PCA(data, dimension = 2):
    #import the basic lib
    from sklearn.decomposition import PCA
    pca = PCA(n_components = dimension)

    Y = data.label

    for i in range(3):
        arrayData = np.array(data.data[i])
        transformedData = pca.fit(arrayData).transform(arrayData)

    colors = ['bastnaesite', 'turquoise', 'darkorange']

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(transformedData[y == i, 0], transformedData[y == i, 1], color=color, alpha=.8, lw=2,
                label=target_name)

    plt.legend()
    plt.show()

    return dict_minerals_PCA

#this function use LDA(linear discriminant Analysis). Similar to PCA
def cal_LDA(dict_minerals_OptParams, dimension = 2):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
    lda = LDA(n_components = dimension)

    dict_minerals_LDA = {}
    X = []

    #tranverse all minerals and LDA them, then plot out results.
    for mineral in sorted(dict_minerals_OptParams.keys()):

        BandList = dict_minerals_OptParams[mineral]

        #merge list parameters in dict[mineral][band] into a larger one 
        XList = []
        for band in sorted(BandList.keys()):
            OptParams_temp = BandList[band]
            numTemp = int(len(OptParams_temp)/3)
            XList_temp = []
            
            #norm the height, width, centers separately. 780,790,798 - [0.x 0.y 0.z]
            for i in range(numTemp):
                listTemp = OptParams_temp[i * numTemp : (i+1)* numTemp]
                listTemp = np.array(listTemp)
                listNorm = (listTemp - listTemp.min()) / (listTemp.max() - listTemp.min())
                XList_temp.extend(listNorm)

            XList.extend(XList_temp)

        transformedList = pd.DataFrame(lda.fit_transform(XList))
        
        dict_minerals_LDA.setdefault(mineral, transformedList)

        plt.scatter(transformedList[0], transformedList[1], label = mineral)

    plt.legend()
    plt.show()

    return dict_minerals_LDA

#this funtion receive a dict that contains all minerals' optimal parameters in all bands. return a dict contains all minerals contains width, height, centers
def transformDict(dict_minerals_OptParams):
    dict = {}

    for mineral in sorted(dict_minerals_OptParams.keys()):
        dict.setdefault(mineral,{})
        dict[mineral].setdefault('width', [])
        dict[mineral].setdefault('height', [])
        dict[mineral].setdefault('center', [])
        dict[mineral].setdefault('yshift',[] )

        # transform core code
        dictMineral = dict_minerals_OptParams[mineral]

        for band in sorted(dictMineral.keys()):
            listParams = dictMineral[band]
            numTemp = int(len(listParams)/3)
            for i in range(3):
                if i == 0:
                    dict[mineral]['height'].extend( listParams[i*numTemp : (i+1)*numTemp])
                elif i ==1:
                    dict[mineral]['width'].extend( listParams[i*numTemp : (i+1)*numTemp])
                elif i ==2:
                    dict[mineral]['center'].extend( listParams[i*numTemp : (i+1)*numTemp])
            dict[mineral]['yshift'].extend([listParams[-1]])

    data = datasetsPCA()
    data.data = [[],[],[]]
    
    data.label = sorted(dict.keys())

    for mineral in sorted(dict.keys()):
        data.data[0].append(dict[mineral]['height'])
        data.data[1].append(dict[mineral]['width'])
        data.data[2].append(dict[mineral]['center'])

    return data
#this function get the optimal fitting parameters of spectra library.(DT's custom REE library.)
if __name__ == '__main__':
    
    filePath = 'data/Envi_spectra_lib/'
    fileName_lib = 'DT_KL_CH_BR_VNIRREE_SpecLib.hdr'
    fileName_initialParams = 'initialParams_Minerals.txt'
    sp_lib = open_spectraData(filePath + fileName_lib)
    
    wavelength = sp_lib.bands.centers
    
    dict_minerals_OptParams = {}
    
    for i in range(len(sp_lib.spectra)):
        # get the spectra reflectance and spectra name from enviLib
        reflectance = sp_lib.spectra[i]
        spectrum = np.array([wavelength, reflectance]).T
        sp_name = sp_lib.names[i]
        
        # get initial parameters of this mineral
        initial_parameters = get_initialParams_fromTXT(filePath + fileName_initialParams, sp_name)
        
        # change the initial width, height
        for band in sorted(initial_parameters.keys()):
            initial_parameters[band]['height'] = [-0.02 for i in range(len(initial_parameters[band]['height']))]

        # use a dict to save optimal params -- Minerals-bands-params
        dict_minerals_OptParams.setdefault(sp_name, {})
        
        # for loop into different band
        for band in sorted(initial_parameters.keys()):
            spectrum_band  = ppm.choose_band(spectrum, initial_parameters,band)

            #change the initial params dict (read from .txt file) into a list that fits multiple Gaussian modeling input params
            initParams_list = []
            initParams_list.extend(initial_parameters[band]['height'])
            initParams_list.extend(initial_parameters[band]['width'])
            initParams_list.extend(initial_parameters[band]['center'])
            initParams_list.extend([initial_parameters[band]['yshift']])

            #least square Gaussian modeling.
            optimal_parameters = MGM.fitting_leastSquare(spectrum_band, initParams_list, maxfev = 30000)

            #save optimal modeling params into the dict 
            dict_minerals_OptParams[sp_name].setdefault(band, optimal_parameters)

            if switch_plotBand:
                fig = plt.figure()
                strTitle = sp_name + '_' + band
                fig.suptitle(strTitle,fontsize = 12)
                plt.plot(spectrum_band[:,0], spectrum_band[:,1])
                OptReflectance = MGM.multi_MGM(spectrum_band[:,0],optimal_parameters)
                plt.plot(spectrum_band[:,0], OptReflectance)
                numTemp = int(len(optimal_parameters)/3)

                #list saves single Gaussian model result.
                listSingleRef = []

                for i in range(numTemp):
                    listSingleRef.append(optimal_parameters[i])
                    listSingleRef.append(optimal_parameters[i + numTemp])
                    listSingleRef.append(optimal_parameters[i + 2 * numTemp])
                    listSingleRef.append(optimal_parameters[-1])
                    
                    singleRef = MGM.multi_MGM(spectrum_band[:,0], listSingleRef)
                    plt.plot(spectrum_band[:,0], singleRef)

                    listSingleRef = []

                #plt.show()
                strTitle = strTitle.replace(':','')
                fig.savefig('output/MultipleMineralFitting/DT custom lib/' + strTitle+ '.jpg')
                plt.close()

    fileName_OptParams = 'OptParams_Minerals.txt'

    # write the optimal parameters result into .txt file
    dictWrite(filePath + fileName_OptParams, dict_minerals_OptParams)

    dict_transformed = transformDict(dict_minerals_OptParams)

    #PCA and other lower dimension methods.
    #PCA give up. because every mineral has different number of parameters. Use the Optimal parameters directly.
    dict_minerals_PCA = cal_PCA(dict_transformed)
        


    
    
    