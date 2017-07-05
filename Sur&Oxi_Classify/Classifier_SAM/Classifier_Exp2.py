import spectral as sp
import numpy as np
import math
import traceback
import os 
from PIL import Image
from glob import glob
import time
start_time = time.time()

#cal the average spectrum of a img. Input a img and return an array (the same format as Spectral lib's )
def cal_aver_SP(img):
    width, height, deepth = img.shape
    sum_SP = 0
    count = 0
    for i in range(width):
        for j in range(height):
            pixel_SP = img[i,j]
            if exclude_BG(pixel_SP):
                continue
            else:
                sum_SP += pixel_SP
                count += 1

    return sum_SP / count
    
#exclude the background pixel, into an array(spectrum) and return T/F, True: background; False: not a background
def exclude_BG(pixel_array):
    if sum(pixel_array) == 0:
        return True
    else:
        return False
    
def choice_input():

    choice = input("use muti data as training data? yes/no \n")
    if choice == 'yes' or choice == 'y':
        return  1
    elif choice == 'no' or choice == 'n':
        return  0
    else:
        print ("your choice is not right, please retry again.\n")
        choice_input()
        
# input training data, maybe this would be a Wizzard ( choose files later)。 return two objects: img_oxido and img_sulfuro, both of them are SP.image_open objects.
# input para use_xx_xx is 0 or 1, if 1, use multiple SP's average as training data.
def input_training_data( use_multi_SP_as_reference = 0, ui = False):

    if ui:
        use_multi_SP_as_reference = choice_input()
        print("%d\n" % use_multi_SP_as_reference)

    if use_multi_SP_as_reference == 1:
        filePath = 'data/'
        files_list_oxi = glob(filePath + 'oxidos/'+"*.hdr")
        files_list_sul = glob(filePath + 'sulfuros/'+'*.hdr')
        
        SP_oxi_all = np.array( [0.0 for i in range(len(sp.open_image(files_list_oxi[0])[0,0]))])
        SP_sul_all = np.array( [0.0 for i in range(len(sp.open_image(files_list_oxi[0])[0,0]))])
        count_oxi = 0
        count_sul = 0
        
        if 'spectrum_average.txt' in os.listdir(filePath + 'oxidos/') and 'spectrum_average.txt' in os.listdir(filePath + 'sulfuros/'):
            file_sp_average = open(filePath + 'oxidos/spectrum_average.txt','r')
            SP_ref_oxi = file_sp_average.read()
            SP_ref_oxi = SP_ref_oxi.replace('[','')
            SP_ref_oxi = SP_ref_oxi.replace(']','')
            SP_ref_oxi = SP_ref_oxi.split()
            file_sp_average.close()
            SP_ref_oxi = np.array(SP_ref_oxi, dtype = np.float64)

            file_sp_average = open(filePath + 'sulfuros/spectrum_average.txt','r')
            SP_ref_sul = file_sp_average.read()
            SP_ref_sul = SP_ref_sul.replace('[','')
            SP_ref_sul = SP_ref_sul.replace(']','')
            SP_ref_sul = SP_ref_sul.split()
            file_sp_average.close()
            SP_ref_sul = np.array(SP_ref_sul, dtype = np.float64)
            return [SP_ref_oxi, SP_ref_sul]
        # get the average SP of all oxi pics.
        for i in range(len(files_list_oxi)):
            try:
                img_oxi = sp.open_image(files_list_oxi[i])
            except Exception as err:
                print("Cannot open your oxido file, confirm your filePath is correct!\n Error info: " + str(err.args))
                exit(0)
            sp_oxi = cal_aver_SP(img_oxi)
            
            try :
                 assert len(SP_oxi_all)  == len(sp_oxi)
            except Exception as err:
                print ('Your oxido input training pics has different bands, please choose again.!\n Error info: ' + str(err.args))
                exit(0)
            
            SP_oxi_all += sp_oxi
            
        
        for i in range(len(files_list_sul)):
            try:
                img_sul = sp.open_image(files_list_sul[i])
            except Exception as err:
                print("Cannot open your suluro file, confirm your filePath is correct!\n Error info: " + str(err.args))
                exit(0)
            sp_sul = cal_aver_SP(img_sul)
            
            try:
                assert len(SP_sul_all) == len(sp_sul)
            except Exception as err:
                print ('Your sulfuro input training pics has different bands, please choose again.!\n Error info: ' + str(err.args))
                exit(0)
                
            SP_sul_all += sp_sul
        
        with open(filePath + 'oxidos/' + 'spectrum_average.txt','w') as f_write_oxi:
            f_write_oxi.write(str(SP_oxi_all / len(files_list_oxi)) )
            f_write_oxi.close()

        with open(filePath + 'sulfuros/' + 'spectrum_average.txt', 'w') as f_write_sul:
            f_write_sul.write(str(SP_sul_all / len(files_list_sul)) )
            f_write_sul.close()

        return [SP_oxi_all / len(files_list_oxi), SP_sul_all / len(files_list_sul)]
        
    elif use_multi_SP_as_reference == 0:
        filePath = 'data/'
        fileName_sul = "sulfuros/EscSulf01_Backside_SWIR_Subset_Masked.hdr"
        fileName_oxi = "oxidos/EscOx01B1_rough_SWIR.hdr"
        
        #two training img. Using the aver their of all pixels
        try:
            img_sulfuro = sp.open_image(filePath+fileName_sul)
            img_oxido = sp.open_image(filePath+fileName_oxi)
        except Exception as err:
            print('Cannot open your training file.\n Error info:' + str(err.args), end = '\n')
            exit(0)
        
        SP_ref_oxido = cal_aver_SP(img_oxido)
        SP_ref_sulfuro = cal_aver_SP(img_sulfuro)
        
    else:
        print("choice(use muti SP or single SP) input seems wrong, program ends. I'm sorry about this.\n")
        exit(0)
    
    return [SP_ref_oxido, SP_ref_sulfuro]
    
def input_testing_data(index = '37', type = 'oxido', check_all = False):

    filePath =  'data/'
    if not check_all :
        num = input("input the index number of your testing oxido file(01-41)\n")

        if type == 'oxido':
            fileName_testing = 'oxidos/EscOx'+ num + 'B1_rough_SWIR.hdr'
        elif type == 'sulfuro':
            fileName_testing = 'sulfuros/EscSulf' + num + '_Backside_SWIR_Subset_Masked.hdr'
    elif check_all:
       if type == 'oxido':
            fileName_testing = 'oxidos/EscOx'+ index + 'B1_rough_SWIR.hdr'
       elif type == 'sulfuro':
            fileName_testing = 'sulfuros/EscSulf' + index + '_Backside_SWIR_Subset_Masked.hdr'

    try:
        img_testing = sp.open_image(filePath + fileName_testing)
    except Exception as err:
        print('Cannot open your testing file.\n Error info:' + str(err.args), end = '\n')
        exit(0)
    return img_testing
    
# spectrum angle mapping, input reference, testing spectrum and deepth (spectrum bands). Return an angle between ref and test SP.
def cal_SP_angle(SP_reference, SP_testing, deepth):
    DownSide1 = 0
    DownSide2 = 0
    UpSide = 0
    for d in range(deepth):
        bandValue_testing = SP_testing[d]
        bandValue_reference = SP_reference[d]
        
        UpSide += bandValue_reference* bandValue_testing
        
        DownSide1 += bandValue_reference**2
        DownSide2 += bandValue_testing**2
    
    angle = UpSide/ (DownSide1**0.5 * DownSide2**0.5)
    
    try:
        angle = math.acos(angle)
    except Exception as err:
        print ('the abs(angle) > 1. \n Error info:'+ err.args, end = '\n')
        exit(0)
    return angle
    
def dataProcess_alg_pass(SP):
    return SP

#tranversing the whole testing img and cal each pixel, then classify it. return [res, accurarcy], the first record the classification info and the later saves accurarcy
# input two ref img, testing img and testing type (Default 1 is testing Oxido, 2 is testing Sul)
def Tranversing(SP_reference1, SP_reference2, img_testing, testingType = 'oxido', dataProcess_alg = dataProcess_alg_pass):
    
    if testingType == 'oxido':
        testingType = 1
    elif testingType == 'sulfuro':
        testingType = 2


    width, height, deepth = img_testing.shape
    deepth = len(SP_reference1)
    #res is a list that would save the classification result, 2 is background, 1 is right, 0 is wrong. 
    res = []
    # the pixel number of background
    count_bg = 0
    count_right = 0
    for i in range(width):
        for j in range(height):
            SP_testing = img_testing[i,j]
                
            # if this pixel is background, res = 2
            if exclude_BG(SP_testing):
                res.append(2)
                count_bg += 1
                continue
            
            # pre-algorithm process data.
            SP_testing = dataProcess_alg(SP_testing)

            # compute spectrum angles.
            angle_ref1 = cal_SP_angle(SP_reference1, SP_testing, deepth)
            angle_ref2 = cal_SP_angle(SP_reference2, SP_testing, deepth)

            # attention please: this is the red mark code, maybe u could add more barriers here.
            # attention please: now ref1 is oxido, ref2 is sulfuro, testing img is a oxido
            if testingType == 1:
                if angle_ref1 < angle_ref2:
                    res.append(1)
                    count_right += 1
                else:
                    res.append(0)
            elif testingType == 2:
                if angle_ref1 > angle_ref2:
                    res.append(1)
                    count_right += 1
                else:
                    res.append(0)
    accurarcy = count_right / (width * height - count_bg)       
    return [res,accurarcy]
            
def show_res(res_list,accurarcy, width, height,filePath, resName, showImg = 0):
    if showImg == 1:
        newImg = Image.new('L',(width,height))
        for i in range(width):
            for j in range(height):
                if res_list[i*height+j] == 0:
                    newImg.putpixel((i,j),123)
                elif res_list[i*height + j] == 1:
                    newImg.putpixel((i,j), 255)
                elif res_list[i*height + j] == 2:
                    newImg.putpixel((i,j), 0)
        newImg.save(filePath + resName, 'bmp')
        newImg.show()
    print('\n%s  accurarcy : %f  Running time: %f \n' % (resName,accurarcy,(time.time() - start_time)) )
    
# output the res. Two methods: accurarcy and image, white pixel is right one and black pixel is wrong.
def main(check_all = False):    

    if check_all :
        filePath = 'data/'
        files_list_oxi = glob(filePath + 'oxidos/'+"*.hdr")
        files_list_sul = glob(filePath + 'sulfuros/'+'*.hdr')
        num_oxi = len(files_list_oxi)
        num_sul = len(files_list_sul)
        
        #accuracy dict, the index is testing files' name and values is a list [single_SP, all_SP]
        acc_dict_oxi = {}
        acc_dict_sul = {}
        

        #check all the oxidos.
        for i in range(1,num_oxi+1):
            if i < 10:
                i = str('0' + str(i))
            else:
                i = str(i)
            img_testing = input_testing_data(index = i, type = 'oxido', check_all = True)

            #single reference check
            SP_ref_oxido, SP_ref_sulfuro = input_training_data(use_multi_SP_as_reference = 0)
            res, accurarcy = Tranversing(SP_ref_oxido, SP_ref_sulfuro, img_testing, 1)
            acc_dict_oxi.setdefault(str(img_testing).split('/')[2].split('_')[0] + '_res.bmp',[])
            acc_dict_oxi[str(img_testing).split('/')[2].split('_')[0] + '_res.bmp'].append(accurarcy)

            #all reference check
            SP_ref_oxido, SP_ref_sulfuro = input_training_data(use_multi_SP_as_reference = 1)
            res, accurarcy = Tranversing(SP_ref_oxido, SP_ref_sulfuro, img_testing, 1)
            acc_dict_oxi[str(img_testing).split('/')[2].split('_')[0] + '_res.bmp'].append(accurarcy)

            #showing the progress
            acc_key = str(img_testing).split('/')[2].split('_')[0] + '_res.bmp'
            print('%s   %f   %f\n' % (acc_key, acc_dict_oxi[acc_key][0],acc_dict_oxi[acc_key][1] ))


        #check all the sulfuros  
        for i in range(num_sul):
            
            index0 = files_list_sul[i].split('Esc')[-1].split('Sulf')[-1][0:2]
            img_testing = input_testing_data(index = index0, type = 'sulfuro', check_all = True)
            #single reference check
            SP_ref_oxido, SP_ref_sulfuro = input_training_data(use_multi_SP_as_reference = 0)
            res, accurarcy = Tranversing(SP_ref_oxido, SP_ref_sulfuro, img_testing, 2)
            acc_dict_sul.setdefault(str(img_testing).split('/')[2].split('_')[0] + '_res.bmp',[])
            acc_dict_sul[str(img_testing).split('/')[2].split('_')[0] + '_res.bmp'].append(accurarcy)

            #all reference check
            SP_ref_oxido, SP_ref_sulfuro = input_training_data(use_multi_SP_as_reference = 1)
            res, accurarcy = Tranversing(SP_ref_oxido, SP_ref_sulfuro, img_testing, 2)
            acc_dict_sul[str(img_testing).split('/')[2].split('_')[0] + '_res.bmp'].append(accurarcy)  

            #showing the progress
            acc_key = str(img_testing).split('/')[2].split('_')[0] + '_res.bmp'
            print('%s   %f   %f\n' % (acc_key, acc_dict_sul[acc_key][0],acc_dict_sul[acc_key][1] ))

        #write the results into txt
        file_res = open(filePath + 'accuracy_result2.txt', 'w')

        for i in acc_dict_oxi.keys():
            file_res.write("%s \t %f \t %f\n" % (i,acc_dict_oxi[i][0],acc_dict_oxi[i][1]))
        for i in acc_dict_sul.keys():
            file_res.write("%s \t %f \t %f\n" % (i,acc_dict_sul[i][0],acc_dict_sul[i][1]))

    else:
        #input reference data.
    
        SP_ref_oxido, SP_ref_sulfuro = input_training_data()
    
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


		
		
		
		

if __name__ == '__main__':
    main()

    
    
    
