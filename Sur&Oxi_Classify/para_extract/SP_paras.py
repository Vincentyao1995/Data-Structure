import spectral as sp
import os
import test_algorithm as ta
import numpy as np
import math
from glob import glob

#define switch
testOxi = 1

#check the data. And got acc txt file.

def dataProcess_alg_pass(SP):
    return SP

# useless, rewriten in test_algorithm
def check(SP_ref_oxido, SP_ref_sulfuro, check_all = 0, dataProcess_alg = dataProcess_alg_pass):
    if check_all == 1:    
        filePath = 'data/'
        files_list_oxi = glob(filePath + 'oxidos/'+"*.hdr")
        files_list_sul = glob(filePath + 'sulfuros/'+'*.hdr')
        num_oxi = len(files_list_oxi)
        num_sul = len(files_list_sul)
    
        #accuracy dict, the index is testing files' name and values is a list [single_SP, all_SP]
        acc_dict_oxi = {}
        acc_dict_sul = {}
    
        #switch
        global testOxi
        #check all the oxidos.
        for i in range(num_oxi):
            if testOxi == 0:
                break 
            index0 = files_list_oxi[i].split('Esc')[-1].split('Ox')[-1][0:2]
            # u could also achieve this by : 
            #
                # if i < 10:
                    # i = str('0' + str(i))
                # else:
                    # i = str(i)
            img_testing = ta.input_testing_data(index = index0, type = 'oxido', check_all = check_all)

            #core alg of check
            # ////////SP_ref_oxido, SP_ref_sulfuro = input_training_data(use_multi_SP_as_reference = 1)
            res, accurarcy = ta.Tranversing(SP_ref_oxido, SP_ref_sulfuro, img_testing, testingType = 'oxido', dataProcess_alg = dataProcess_alg)
            acc_dict_oxi.setdefault(str(img_testing).split('/')[2].split('_')[0] + '_res.bmp',[])
            acc_dict_oxi[str(img_testing).split('/')[2].split('_')[0] + '_res.bmp'].append(accurarcy)

            #showing the progress
        
            acc_key = str(img_testing).split('/')[2].split('_')[0] + '_res.bmp'
            print('%s   %f   \n' % (acc_key, acc_dict_oxi[acc_key][0] ))


        #check all the sulfuros  
        for i in range(num_sul):
        
            index0 = files_list_sul[i].split('Esc')[-1].split('Sulf')[-1][0:2] # after split('Esc'), u got a list containing only one element.... x[0] == x[-1]
            img_testing = ta.input_testing_data(index = index0, type = 'sulfuro', check_all = check_all)
        
            #core alg of check
            # /////SP_ref_oxido, SP_ref_sulfuro = input_training_data(use_multi_SP_as_reference = 0)
            res, accurarcy = ta.Tranversing(SP_ref_oxido, SP_ref_sulfuro, img_testing,  testingType = 'sulfuro', dataProcess_alg = dataProcess_alg)
            acc_dict_sul.setdefault(str(img_testing).split('/')[2].split('_')[0] + '_res.bmp',[])
            acc_dict_sul[str(img_testing).split('/')[2].split('_')[0] + '_res.bmp'].append(accurarcy)

        

            #showing the progress
            acc_key = str(img_testing).split('/')[2].split('_')[0] + '_res.bmp'
            print('%s   %f   \n' % (acc_key, acc_dict_sul[acc_key][0] ))

        #write the results into txt
        file_res = open(filePath + '3paras_accuracy_non_normalized.txt', 'w')
        file_res.write('fileName \t \t Accuracy\n')
        for i in acc_dict_oxi.keys():
            file_res.write("%s \t %f\n" % (i,acc_dict_oxi[i][0]))
        for i in acc_dict_sul.keys():
            file_res.write("%s \t %f\n" % (i,acc_dict_sul[i][0]))
    elif check_all == 0:
    
        #input testing data
        img_testing = input_testing_data()
    
        # tranversing the img and cal spectral angle between testImg and refImg. 
        #Input: testing img and reference img.
        if 'Sulf' in str(img_testing): 
            res, accurarcy = Tranversing(SP_ref_oxido,SP_ref_sulfuro, img_testing, 2)
        
        else:
            res, accurarcy = Tranversing(SP_ref_oxido,SP_ref_sulfuro, img_testing, 1)

        width, height, deepth = img_testing.shape
    
        resName = str(img_testing).split('/')[2].split('_')[0] + '_res.bmp'
        filePath = 'data/'
        show_res(res,accurarcy, width, height, filePath, resName, showImg = 0)


# input an para dict, return a list containing all its data. para_dict['band1']['AA'] = ...
#para_list = [1,2,3,0.1.....]
def dict_to_list(para_dict):
    
    para_list = []
    for i in para_dict.keys() :
        for j in para_dict[i].keys():
            para_list.append(para_dict[i][j])
            
    return para_list

#input three arrays and normalize them, return three normalized arrays.
def normalize(arr1,arr2, arr3):

    arr_length = len(arr1)
    assert len(arr1) == len(arr2) and len(arr2) == len(arr3), 'length of your three arrays to be normalized is not equal! \n'
    for i in range(arr_length):
        value_list = [arr1[i], arr2[i], arr3[i]]
        max_value = max(value_list)
        min_value = min(value_list)
        if max_value == min_value:
            arr1[i] = 1
            arr2[i] = 1
            arr3[i] = 1
            continue
        arr1[i] = ( arr1[i] - min_value ) / (max_value - min_value)
        arr2[i] = ( arr2[i] - min_value ) / (max_value - min_value)
        arr3[i] = ( arr3[i] - min_value ) / (max_value - min_value)

    return [arr1, arr2, arr3]
            
#load average sulfuros data, return a spectrum array. 
def load_training_SP(type = 'sulfuro'):
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
        file_temp = sp.open_image(filePath + type + 's/' + fileNameList[1])
        file_temp.close()
        pass
    sp.open_image(filePath + fileName)

# input auto_set, default 1 and set the ABPs (absorption bands) are set mannually
# type controls the ABP bands' difference between sulf and oxi. (tho no differences now.)
# sp is a spectrum.
def choose_ABP_bands(sp, auto_set = 1, type = 'sulfuro', choose_band = [1,1,1]):

    if auto_set == 1:
        
        #sp's sampling interval is around 6.3 um, not continuous. 928.08 - 2530.1500
        #attention, need to set the bands automatically.
        ABP_bands_sulf = []
        ABP_bands_oxi = []
        for i in range(len(choose_band)):
            if choose_band[i] == 1 and i==0:
                ABP_bands_sulf.append(sp[51:91])  # see at band excel, starting from 1;   52-92  1250um-1500um
                ABP_bands_oxi.append(sp[51:91])
                continue
            if choose_band[i] == 1 and i==1:
                ABP_bands_sulf.append(sp[138:171])  # 1800:2000 ,139-172
                ABP_bands_oxi.append(sp[138:171])
                continue
            if choose_band[i] == 1 and i==2:
                ABP_bands_sulf.append(sp[186:211])  # 2100:2250 , 187-212 
                ABP_bands_oxi.append(sp[186:211])
                continue
        
        
        
        
        
        
        if type == 'sulfuro':
            return ABP_bands_sulf
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
        right_half_point = 0
        left_half_point = 0
        maxRef = max(ABP_bands_SP[i])
        for j in range(len(ABP_bands_SP[i])-1):
            midRef = (minRef + maxRef) / 2
            if ABP_bands_SP[i][j] >= midRef and ABP_bands_SP[i][j+1] < midRef:
                left_half_point = j
            if ABP_bands_SP[i][j] <= midRef and ABP_bands_SP[i][j+1] > midRef :
                right_half_point = j
        AW = abs(right_half_point - left_half_point) * step

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


def SP_paras(SP_array, type = 'sulfuro', choose_band = [1,1,1]):

    ABP_bands = choose_ABP_bands(SP_array, type = type, choose_band = choose_band)
    para_dict = cal_SP_paras(ABP_bands)
    return para_dict
    
    
if __name__ == '__main__':
    
    # 1. load sp data from img.
    sp_average_sul = load_training_SP(type = 'sulfuro')    
    # 2. choose interesting area (absorption bands) in a spectrum
    sul_ABP_bands = choose_ABP_bands(sp_average_sul)
    # 3. Cal SP paras of this areas, and got the set of features.
    para_dict = cal_SP_paras(sul_ABP_bands)
    print("end\n")
    
    