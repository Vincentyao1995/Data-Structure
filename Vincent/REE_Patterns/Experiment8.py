
#this script is do Gaussian modeling for multiple minerals and output the paramters like DT's method. 

import spectral.io.envi as envi
import ModifiedGaussianModel as MGM
import numpy as np
import pre_processing_mineral as ppm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

switch_plotBand = 1



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

#this function input a dict including optimal parameters from multiple minerals. return the PCA result.
def cal_PCA(dict_minerals_OptParams, dimension = 2):
    #import the basic lib
    from sklearn.decomposition import PCA
    pca = PCA(n_components = dimension)

    dict_minerals_PCA = {}

    #tranverse all minerals and PCA them, then plot them out.
    for mineral in sorted(dict_minerals_OptParams.keys()):

        XList = dict_minerals_OptParams[mineral]
        X_norm = (XList - XList.min()) / (XList.max() - XList.min())
        transformedList = pd.DataFrame(pca.fit_transform(X_norm))
        
        #save the PCA results into a list.
        dict_minerals_PCA.setdefault(mineral, transformedList)

        plt.scatter(transformedList[0], transformedList[1], label = mineral)

    plt.legend()
    plt.show()

    return dict_minerals_PCA

#this function use LDA(linear discriminant Analysis). Similar to PCA
def cal_LDA(dict_minerals_OptParams, dimension = 2):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
    lda = LDA(n_components = dimension)

    dict_minerals_LDA = {}

    #tranverse all minerals and LDA them, then plot out results.
    for mineral in sorted(dict_minerals_OptParams.keys()):

        X = dict_minerals_OptParams[mineral]
        X_norm  = (X - X.min()) / (X.max() - X.min())
        transformedList = pd.DataFrame(lda.fit_transform(X_norm, y ))
        
        dict_minerals_LDA.setdefault(mineral, transformedList)

        plt.scatter(transformedList[0], transformedList[1], label = mineral)

    plt.legend()
    plt.show()

    return dict_minerals_LDA


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

    #PCA and other lower dimension methods.
    dict_minerals_PCA = cal_PCA(dict_minerals_OptParams)
    #time to debug PCA and plot results out. 
    


    
    
    