import spectral as sp
import os
import Classifier_Exp2 as cl2
import numpy as np
import math
from glob import glob


#load average sulfuros data, return a spectrum array. 
#attention: need to extract paras of every pic, tranversing these pic and extrac them. And don't know access of muti sp -traning and extracting yet. Possible solution: [sp_array1, sp_array2, sp_array3].
def load_data(type = 'sulfuro'):
    filePath = 'data/'
    # fileName = 'sulfuros/'

    fileNameList =  os.listdir(filePath + type +'s/') #  glob(filePath + 'sulfuros/' + '*.hdr')
    
    fileName_aver = 'spectrum_average.txt'
    if fileName_aver in fileNameList :
        file_temp = open(filePath + type + 's/' + fileName_aver)
        sp_average = file_temp.read()
        sp_average = sp_average.replace('[', '')
        sp_average = sp_average.replace(']', '')
        sp_average = sp_average.split()
        file_temp.close()
        return np.array(sp_average ,dtype = np.float)
    else:
        #read all the hdr's and got the average spectrum of all the data. attention: maybe we could use the def in 'Classifier_Exp2'
        #see? this is so easy in python.
        #cl2.cal_aver_SP()
        file_temp = sp.open_image(filePath + type + 's/' + fileNameList[1])
        file_temp.close()
        pass
    sp.open_image(filePath + fileName)

# input auto_set, default 1 and set the ABPs (absorption bands) are set mannually
def choose_ABP_bands(sp, auto_set = 1, type = 'sulfuro'):
    if auto_set == 1:
        
        #sp's sampling interval is around 6.3 um, not continuous. 928.08 - 2530.1500
        #attention, need to set the bands automatically.
        ABP_bands_sulf = []
        ABP_bands_sulf.append(sp[51:91])  # see at band excel, starting from 1;   52-92  1250um-1500um
        ABP_bands_sulf.append(sp[138:171])  # 1800:2000 ,139-172
        ABP_bands_sulf.append(sp[186:211])  # 2100:2250 , 187-212 
        
        if type == 'sulfuro':
            return ABP_bands_sulf
        
        ABP_bands_oxi = []
        ABP_bands_oxi.append(sp[51:91])
        ABP_bands_oxi.append(sp[138:171])
        ABP_bands_oxi.append(sp[186:211])
        
        if type == 'oxido':
            return ABP_bands_oxi
        
    else:
        # maybe Wizzard here. or design some auto-detect ABP alg
        pass
        
#input absorption bands(some specific area) of a spectrum, and return these bands' paras	
def cal_SP_paras(ABP_bands_SP):
    
    ####################################attention: this part is manually, manually, manually!!!######################################
    
    #the key of band_index is 0-255, value is wavelength
    band_index = {}

    step = int ((2530.15-928.08)/ 256)
    for i in range(256):
        band_index.setdefault(i, 928.08 + i* step)

    bands_init = [] 
    bands_init.append(51)
    bands_init.append(138)
    bands_init.append(186)
    bands_end = []
    bands_end.append(91)
    bands_end.append(171)
    bands_end.append(211)
    #################################################################################################################################



    

    #para_dict is a dict containing absorption info of a spectrum. para_dict[band1][AD] = xxxx,  para_dict[band2][SAI]
    para_dict = {}
    for i in range(len(ABP_bands_SP)):
        # key_temp = 'band' + str(ABP_bands_SP[i][0]) + '-' str(ABP_bands_SP[i][1])  this is band1250-1500
        key_temp = 'band' + str(i+1) # this is band1, band2
        
        para_dict.setdefault(key_temp, {})
        # set initial value to para_dict
        para_dict[key_temp].setdefault('AP', 0)
        para_dict[key_temp].setdefault('AD', 0)
        para_dict[key_temp].setdefault('AW', 0)
        para_dict[key_temp].setdefault('AA', 0)
        para_dict[key_temp].setdefault('AS', 0)
        para_dict[key_temp].setdefault('SAI', 0)
    
        ####################################cal paras of ABP bands.###########################################

        #cal AP, ABP position.
        minRef = min(ABP_bands_SP[i])
        minRef_index = -1
        for j in range(len(ABP_bands_SP[i])):
            if ABP_bands_SP[i][j] == minRef:
                minRef_index = j
        AP = band_index[  j + bands_init[i]   ]

        #cal AD, ABP deepth
        AD = 1 - minRef

        #cal AW, ABP width
        maxRef = max(ABP_bands_SP[i])
        for j in range(len(ABP_bands_SP[i])-1):
            midRef = (minRef + maxRef) / 2
            if ABP_bands_SP[i][j] >= midRef and ABP_bands_SP[i][j+1] < midRef:
                left_half_point = j
            if ABP_bands_SP[i][j] <= midRef and ABP_bands_SP[i][j+1] > midRef :
                right_half_point = j
        AW = (right_half_point - left_half_point) * step

        #cal AA
        AA = 0.0
        for j in range(len(ABP_bands_SP[i])):
            AA += 1 - ABP_bands_SP[i][j]

        #cal AS
        leftArea = 0.0
        rightArea = 0.0
        for j in range(len(ABP_bands_SP[i])):
            if j < minRef_index:
                leftArea += 1 - ABP_bands_SP[i][j]
            else:
                rightArea += 1 - ABP_bands_SP[i][j]
        AS = math.log1p(rightArea/ leftArea )

        #cal SAI
        d = (AP - band_index[bands_init[i]])/(band_index[bands_end[i]] - band_index[bands_init[i]])
        upside = ( d * ABP_bands_SP[i][-1] )+  ( (1-d) * ABP_bands_SP[i][0] )
        downside = minRef
        SAI = upside / downside
        #########################################################################################################

        para_dict[key_temp]['AP'] = AP
        para_dict[key_temp]['AD'] = AD
        para_dict[key_temp]['AW'] = AW
        para_dict[key_temp]['AA'] = AA
        para_dict[key_temp]['AS'] = AS
        para_dict[key_temp]['SAI'] = SAI
        # for end.

    return para_dict

if __name__ == '__main__':
    
    # 1. load sp data from img.
    sp_average_sul = load_data(type = 'sulfuro')    
    # 2. choose interesting area (absorption bands) in a spectrum
    sul_ABP_bands = choose_ABP_bands(sp_average_sul)
    # 3. Cal SP paras of this areas, and got the set of features.
    para_dict = cal_SP_paras(sul_ABP_bands)
    print("end\n")
    
    