import pandas as pd
import test_algorithm as ta
import Classifier_SAM as SAM
import SP_paras
import numpy as np

filePath = 'C:/Users/acer-pc/Desktop/Mitacs/_data/'
fileName = 'Escondida_Combined.xlsx'

dataFrame = pd.read_excel(filePath + fileName, sheetname = 'python')
dataFrame = dataFrame.fillna(0.0)

fileNames = list(dataFrame.index)
colNames = list(dataFrame.columns)

amount_Kaolinite = list(dataFrame[colNames[1]])
amount_Gypsum = list(dataFrame[colNames[12]])
amount_Alunite = list(dataFrame[colNames[14]])
amount_Antlerite = list(dataFrame[colNames[16]])




mineral_dict = {}
mineral_dict.setdefault('AA',[])
mineral_dict.setdefault('AS',[])
mineral_dict.setdefault('AD',[])
mineral_dict.setdefault('SAI',[])
mineral_dict.setdefault('AP',[])
mineral_dict.setdefault('AW',[])

# got the paras from all (containing minerals) pics
for i in range(len(fileNames)):
    if i < 9:
        index = '0' + str(i+1)
    else:
        index = str(i+1)
    #open the image and got the average sp of this image.
    if 'Ox' in fileNames[i]:	
        image = ta.load_image(type = 'oxido', index = index)
    if 'Sulf' in fileNames[i]:
        image = ta.load_image(type = 'oxido', index = index)
    sp_average = SAM.cal_aver_SP(image)
    
    para_dict = SP_paras.SP_paras(sp_average,choose_band = [1,0,0])
    para_list = SP_paras.dict_to_list(para_dict)
    for key in mineral_dict.keys():
        for key_temp in para_dict.keys():
            mineral_dict[key].append(para_dict[key_temp][key])
    
# cal correlation between mineral_dict (paras) and amount_minerals


coe_Kaolinite = {}
coe_Gypsum = {}
coe_Alunite = {}
coe_Antlerite = {}

for key in sorted(mineral_dict.keys()):
    coe_temp = np.corrcoef(mineral_dict[key], amount_Alunite)
    coe_Kaolinite.setdefault(key,coe_temp)
    

    coe_temp = np.corrcoef(mineral_dict[key], amount_Antlerite)
    coe_Antlerite.setdefault(key,coe_temp)


    coe_temp = np.corrcoef(mineral_dict[key], amount_Gypsum)
    coe_Gypsum.setdefault(key,coe_temp)
    
    coe_temp = np.corrcoef(mineral_dict[key], amount_Kaolinite)
    coe_Alunite.setdefault(key,coe_temp)


nameStr = ['Kaolinite','Antlerite','Gypsum','Alunite']

    
file_out = open('data/corrcoef.txt','w')
file_out.write('\t\t\t AA \t AD \t AP \t AS \t AW \t SAI\n')    
for i in range(4):
    if not i == 0: 
        file_out.write('\n') 
    file_out.write(nameStr[i] + '\t')
    for key in sorted(mineral_dict.keys()):
        if i == 0:
            file_out.write('\t%f\t' % coe_Kaolinite[key][0][1])
        if i == 1:
            file_out.write('\t%f\t' % coe_Gypsum[key][0][1])
        if i == 2:
            file_out.write('\t%f\t' % coe_Alunite[key][0][1])
        if i == 3:
            file_out.write('\t%f\t' % coe_Antlerite[key][0][1])
       

print('done!\n')