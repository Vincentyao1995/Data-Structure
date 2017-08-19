import spectral.io.envi as envi
import ModifiedGaussianModel as MGM
import numpy as np

#this script is do Gaussian modeling for multiple minerals and output the paramters like DT's method. 

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
def plotOut(dict_minerlas_OptParams):

    return 0

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
        optimal_parameters = MGM.fitting_leastSquare(spectrum, initial_parameters)
        #time to extract band info, and form then info to a list to input Gaussian modeling.


        dict_minerals_OptParams.setdefault(sp_name, optimal_parameters)#attention, search how to plot out high dimension data.
        
    plotOut(dict_minerals_OptParams)    
            
    
    