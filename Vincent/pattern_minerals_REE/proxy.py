import test_algorithm as ta
import SP_paras
import pandas as pd
import numpy as np
from glob import glob
import os

threshold_noise = 0.005
discard_paras = 1

#open an image, attention: Tranversing all the pics later.

bands = [[1350,1500]]



#attention, maybe u could input all the bands and cal proxy, then cal corrcoef by code, at last cal corrcoef so that know which band is most related to amount.      for band in bands[0:2500] df_proxy = cal_proxy_all_files(band) ; corrcoef(df_proxy, amount)  And output corrcoef and maybe filter it automatically

#attention: time to change choose_band function, and seems mutiple band I don't consider, so that the structure of dataFrame is not right, maybe calculate proxy is also wrong. Maybe write band after one band is done. change the cal_proxy. para_dict['bandx']['AA'] 


#this funtion input the band you choose, and return a dateFrame that saves the proxy of all the pics
def cal_proxy_all_file(bands = bands):

    filePath = 'data/'
    files_list_oxi = glob(filePath + 'oxidos/'+"*.hdr")
    name_list_oxi = [name for name in os.listdir(filePath + 'oxidos/') if name.endswith('.hdr')]
    name_list_sulf = [name for name in os.listdir(filePath + 'sulfuros/') if name.endswith('.hdr')]
    files_list_sulf = glob(filePath + 'sulfuros/'+'*.hdr')
    num_oxi = len(files_list_oxi)
    num_sulf = len(files_list_sulf)

    df_proxy = None

    for i in range(num_oxi):
        image = ta.load_image(filePath = files_list_oxi[i])
        proxy_dict = cal_proxy(image,bands = bands)
        
        #initial the dateFrame that saves all files' proxies.
        if i ==0:
            df_proxy = pd.DataFrame(np.zeros((len(name_list_oxi)+len(name_list_sulf), len(proxy_dict.keys()))),columns = proxy_dict.keys(),index = name_list_oxi + name_list_sulf)
 
        #write proxy to dateframe.
        for key in proxy_dict:
            df_proxy[key][name_list_oxi[i]] = proxy_dict[key]

        # to show the progress
        print('%s done!' % name_list_oxi[i])

    for i in range(num_sulf):
        image = ta.load_image(filePath = files_list_sulf[i])
        proxy_dict = cal_proxy(image,bands = bands)

        #write proxy to dataframe
        for key in proxy_dict:
            df_proxy[key][name_list_sulf[i]] = proxy_dict[key]

        # to show the progress
        print('%s done!' % name_list_sulf[i])

    return df_proxy

#this function input an image, and return its proxy, as a dict, containing different proxy, AA_proxy, SAI_proxy
def cal_proxy(image,bands = bands):
    global discard_paras
    global threshold_noise

    proxy_dict = {}
    proxy_dict.setdefault('AA', 0.0)
    proxy_dict.setdefault('AD', 0.0)
    proxy_dict.setdefault('AW', 0.0)
    proxy_dict.setdefault('AS', 0.0)
    proxy_dict.setdefault('AP', 0.0)
    proxy_dict.setdefault('SAI', 0.0)
    
    width, height, deepth = image.shape
    
    count_bg = 0
    proxy_pop_mark = 0
    for i in range(width):
        for j in range(height):
            sp_pixel = image[i,j]

            #ignore the background pixel
            if ta.exclude_BG(sp_pixel):
                count_bg += 1
                continue        
            
            para_dict = SP_paras.SP_paras(sp_pixel,wavelength = bands)
            para_dict = SP_paras.dict_to_dataFrame(para_dict)
            if discard_paras:
                para_dict.pop('AA')
                para_dict.pop('AP')
                para_dict.pop('AW')
                if proxy_pop_mark == 0:
                    proxy_dict.pop('AA')
                    proxy_dict.pop('AP')
                    proxy_dict.pop('AW')
                    proxy_pop_mark = 1
            # sum paras in para_dict so that we get proxies. 
            for key in para_dict:
                if para_dict[key] > threshold_noise:
                    proxy_dict[key] += para_dict[key]
    # the number of rock pixels  =  total - background pixel
    count_rocks = height* width - count_bg

    #cal proxy, summation of all pixels' paras/ pixels' number
    for key in proxy_dict:
        proxy_dict[key] /= count_rocks

    return proxy_dict

#this function accept a string: fileOutName, and output the proxy of all file into this file.
def output_proxy(fileOutName = 'proxy_all_file.txt', bands = bands):
    df_proxy = cal_proxy_all_file(bands = bands)
    file_out = open('data/' + fileOutName,'w')
    df_proxy.to_string(file_out)
    file_out.write('\n\t\tbands: %d - %d\n' % (bands[0][0],bands[0][1]))
    print('all done!\n')

#attention
def corrcoef():
    pass