
#this script is do Gaussian modeling for multiple minerals and output the paramters like DT's method. 

import spectral.io.envi as envi
import ModifiedGaussianModel as MGM
import numpy as np
import pre_processing_mineral as ppm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

switch_plotBand = 1

# input the filePath and return a spectral.io.envi class including many info.
def open_spectraData(filePath):
    sp_lib = envi.open(filePath)

    return sp_lib

# input the filePath and mineral name u want, return initial Gaussian modeling parameters.
def get_initialParams(filePath, sp_name):
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
                

    return dict_mineral_initParams[sp_name]

#input a dict that contains different minerals' optimal fitting parameters and plot them out as the DT's. 
# also, output the optimal multiple Gaussian modeling results with the original curves
def plotOut(dict_minerlas_OptParams, sp_lib):
    #list saves height weight and center, attention, time to plotout DT's result. But only got 6.
    listHWC = []
    listHWC.append(list(optimal_parameters[0:numTemp]))
    listHWC.append(list(optimal_parameters[numTemp:numTemp*2]))
    listHWC.append(list(optimal_parameters[numTemp*2:-1]))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(listHWC[0], listHWC[1], listHWC[2])
    ax.set_xlabel('Height')
    ax.set_ylabel('Width')
    ax.set_zlabel('Center')
    return 0
    



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


if __name__ == '__main__':
    
    filePath = 'data/'
    fileName_lib = 'SpectraForAbsorptionFitting.hdr'
    fileName_initialParams = 'initialParams_Minerals.txt'
    sp_lib = open_spectraData(filePath + fileName_lib)
    
    wavelength = sp_lib.bands.centers
    
    dict_minerals_OptParams = {}
    
    for i in range(len(sp_lib.spectra)):
        # get the spectra reflectance and spectra name from enviLib
        reflectance = sp_lib.spectra[i]
        spectrum = np.array([wavelength, reflectance]).T
        sp_name = sp_lib.names[i].split('_')[0]
        
        # get initial parameters of this mineral
        initial_parameters = get_initialParams(filePath + fileName_initialParams, sp_name)
        
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



                plt.show()
    
    fileName_OptParams = 'OptParams_Minerals.txt'

    # write the optimal parameters result into .txt file
    dictWrite(filePath + fileName_OptParams, dict_minerals_OptParams)

    # plot out the optimal parameters in grid
    plotOut(dict_minerals_OptParams, sp_lib)


    
    
    