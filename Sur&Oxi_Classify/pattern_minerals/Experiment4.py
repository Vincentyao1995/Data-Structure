import pandas as pd
import test_algorithm as ta
import Classifier_SAM as SAM
import SP_paras
import numpy as np

# Switches ~~~~~~~~~~~~~~~
debugging = 0
outputParas = 1

def output_paras(paras_dict):
    filePath = 'data/'
    fileName = 'averageSP_allPics_paras.txt'
    file_output = open(filePath+fileName, 'w')
    file_output.write("fileName\t\t\tAA\t\t\tAD\t\t\tAP\t\t\tAS\t\t\tAW\t\t\tSAI\n")
    for i in range(len(paras_dict['AA'])):
        file_output.write("%d\t\t\t\t%f\t\t\t%f\t\t\t%f\t\t\t%f\t\t\t%f\t\t\t%f\n" % (i+1,paras_dict['AA'][i],paras_dict['AD'][i],paras_dict['AP'][i],paras_dict['AS'][i],paras_dict['AW'][i],paras_dict['SAI'][i]))
        # or for key in sorted(paras_dict.keys()): file.write("%f\t\t\t" % paras_dict[key])
    
def corrcoef_para_amount(choose_band = [1,0,0], file_out_name = 'paras_bands.txt', read_Rietv = 0):
    global outputParas
    filePath = 'C:/Users/acer-pc/Desktop/Mitacs/_data/'
    fileName = 'python_kaolin.xlsx'

    dataFrame = pd.read_excel(filePath + fileName, sheetname = 'Sheet1')
    dataFrame = dataFrame.fillna(0.0)

    fileNames = list(dataFrame.index)
    colNames = list(dataFrame.columns)

    if read_Rietv == 1:
        amount_Kaolinite = list(dataFrame[colNames[1]])
        amount_Gypsum = list(dataFrame[colNames[12]])
        amount_Alunite = list(dataFrame[colNames[14]])
        amount_Antlerite = list(dataFrame[colNames[16]])
    else:
        amount = []
        for i in colNames:
            amount.append(list(dataFrame[i]))



    mineral_dict = {}
    mineral_dict.setdefault('AA',[])
    mineral_dict.setdefault('AS',[])
    mineral_dict.setdefault('AD',[])
    mineral_dict.setdefault('SAI',[])
    mineral_dict.setdefault('AP',[])
    mineral_dict.setdefault('AW',[])

    
    # got the paras from all (containing minerals) pics
    for i in range(len(fileNames)):
        #open the image and got the average sp of this image.
        if 'Ox' in fileNames[i]:	
            index = fileNames[i].split('Ox')[-1].split('B')[0]
            if float(index) < 10:
                index = '0' + index
            image = ta.load_image(type = 'oxido', index = index)

        if 'Sulf' in fileNames[i]:
            index = fileNames[i].split('-')[-1]
            if float(index) < 10:
                index = '0' + index
            image = ta.load_image(type = 'sulfro', index = index)
        sp_average = SAM.cal_aver_SP(image)
    
        para_dict = SP_paras.SP_paras(sp_average,choose_band = choose_band)
        para_list = SP_paras.dict_to_list(para_dict)
        for key in mineral_dict.keys():
            for key_temp in para_dict.keys():
                mineral_dict[key].append(para_dict[key_temp][key])
    if outputParas == 1:
        output_paras(mineral_dict)
        print('paras output done!\n')
        return 0
    # cal correlation between mineral_dict (paras) and amount_minerals
    coe_Kaolinite = {}
    coe_Gypsum = {}
    coe_Alunite = {}
    coe_Antlerite = {}
    data_temp = np.zeros((len(colNames),len(mineral_dict.keys())))
    df_corrcoef = pd.DataFrame(data_temp,index = colNames, columns = list(mineral_dict.keys()))#attention, use dataFrame to save corrcoef 

    for key in sorted(mineral_dict.keys()):
        if read_Rietv == 1:
            coe_temp = np.corrcoef(mineral_dict[key], amount_Alunite)
            coe_Kaolinite.setdefault(key,coe_temp[0][1])
    

            coe_temp = np.corrcoef(mineral_dict[key], amount_Antlerite)
            coe_Antlerite.setdefault(key,coe_temp[0][1])


            coe_temp = np.corrcoef(mineral_dict[key], amount_Gypsum)
            coe_Gypsum.setdefault(key,coe_temp[0][1])
    
            coe_temp = np.corrcoef(mineral_dict[key], amount_Kaolinite)
            coe_Alunite.setdefault(key,coe_temp[0][1])
        else:
            for i in range(len(colNames)):
                coe_temp = np.corrcoef(mineral_dict[key], amount[i])
                df_corrcoef[key][i] = coe_temp[0][1]

    if read_Rietv == 1:
        nameStr = ['Kaolinite','Antlerite','Gypsum\t','Alunite\t']
    
    #write corrcoef to files
    filePath = 'data/'
    if read_Rietv == 1:
        if choose_band[0] == 1:
            file_out = open(filePath + file_out_name, 'w')
            file_out.write('\t\t\t\tAA\t\t\t\tAD\t\t\t\tAP\t\t\t\tAS\t\t\t\tAW\t\t\t\tSAI')    
            file_out.write('\nABP Band1: 1250nm-1500nm  -------------------------------------------------------------------------------\n')
            for i in range(4):
                if not i == 0: 
                    file_out.write('\n') 
                file_out.write(nameStr[i] + '\t')
                for key in sorted(mineral_dict.keys()):
                    if i == 0:
                        file_out.write('\t%f\t' % coe_Kaolinite[key])
                    if i == 1:
                        file_out.write('\t%f\t' % coe_Gypsum[key])
                    if i == 2:
                        file_out.write('\t%f\t' % coe_Alunite[key])
                    if i == 3:
                        file_out.write('\t%f\t' % coe_Antlerite[key])
            file_out.write('\nABP Band2: 1800nm-2000nm  -------------------------------------------------------------------------------\n')
        else :
            file_out = open(filePath + file_out_name,'a')
            for i in range(4):
                if not i == 0: 
                    file_out.write('\n') 
                file_out.write(nameStr[i] + '\t')
                for key in sorted(mineral_dict.keys()):
                    if i == 0:
                        file_out.write('\t%f\t' % coe_Kaolinite[key])
                    if i == 1:
                        file_out.write('\t%f\t' % coe_Gypsum[key])
                    if i == 2:
                        file_out.write('\t%f\t' % coe_Alunite[key])
                    if i == 3:
                        file_out.write('\t%f\t' % coe_Antlerite[key])
            file_out.write('\nABP Band3: 2100nm-2250nm  --------------------------------------------------------------------------------\n')
        print('done!\n')
    else:
        if choose_band[0] == 1:
            file_out = open(filePath + file_out_name, 'w')
            file_out.write('\t\t\t\tAA\t\t\t\tAD\t\t\t\tAP\t\t\t\tAS\t\t\t\tAW\t\t\t\tSAI')    
            file_out.write('\nABP Band1: 1250nm-1500nm  -------------------------------------------------------------------------------\n')
            for i in colNames:
                if not i == 0: 
                    file_out.write('\n') 
                file_out.write(i + '\t')
                for key in sorted(mineral_dict.keys()):
                    file_out.write('\t%f\t' % df_corrcoef[key][i])
            print('band1: done!\n')
            file_out.write('\nABP Band2: 1800nm-2000nm  -------------------------------------------------------------------------------\n')
        else :
            file_out = open(filePath + file_out_name,'a')
            for i in colNames:
                if not i == 0: 
                    file_out.write('\n') 
                file_out.write(i + '\t')
                for key in sorted(mineral_dict.keys()):
                    file_out.write('\t%f\t' % df_corrcoef[key][i])

            file_out.write('\nABP Band3: 2100nm-2250nm  --------------------------------------------------------------------------------\n')
            print('band2 or 3 done!\n')
        
   

if __name__ == '__main__':
    corrcoef_para_amount(choose_band = [1,0,0])
    #corrcoef_para_amount(choose_band = [0,1,0])
    #corrcoef_para_amount(choose_band = [0,0,1])
    print('all done!\n')