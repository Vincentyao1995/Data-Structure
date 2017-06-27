import spectral as sp
import numpy as np
import math
import traceback
from PIL import Image

#attention: need to change res's name and 

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

# input training data, maybe this would be a Wizzard ( choose files later)
def input_traning_data():
    pass

def input_testing_data():
    pass

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
    
#tranversing the whole testing img and cal each pixel, then classify it. return [res, accurarcy], the first record the classification info and the later saves accurarcy
# input two ref img, testing img and testing type (Default 1 is testing Oxido, 2 is testing Sul)
def Tranversing(img_reference1, img_reference2, img_testing, testingType = 1):
    aver_reference1 = cal_aver_SP(img_reference1)
    aver_reference2 = cal_aver_SP(img_reference2)
    
    width, height, deepth = img_testing.shape
    
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
            
            SP_reference1 = aver_reference1
            SP_reference2 = aver_reference2
            angle_ref1 = cal_SP_angle(SP_reference1, SP_testing, deepth)
            angle_ref2 = cal_SP_angle(SP_reference2,SP_testing, deepth)

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
            
def show_res(res_list,accurarcy, width, height,filePath,resName):
    newImg = Image.new('L',(width,height))
    for i in range(width):
        for j in range(height):
            if res_list[i*height+j] == 0:
                newImg.putpixel((i,j),123)
            elif res_list[i*height + j] == 1:
                newImg.putpixel((i,j), 255)
            elif res_list[i*height + j] == 2:
                newImg.putpixel((i,j), 0)
    

    print('\n your accurarcy is : %f \n' % accurarcy )
    newImg.save(filePath + resName, 'bmp')
    newImg.show()
# output the res. Two methods: accurarcy and image, white pixel is right one and black pixel is wrong.
if __name__ == '__main__':
    
    #open the file
    input_traning_data()
    filePath = 'data/'
    fileName_sul = "sulfuros_sub/EscSulf01_Backside_SWIR_Subset_Masked.hdr"
    fileName_oxi = "oxidos/EscOx01B1_rough_SWIR.hdr"
    fileName_testing = "oxidos/EscOx37B1_rough_SWIR.hdr"
    #two training img. Using the aver their of all pixels
    try:
        img_sulfuro = sp.open_image(filePath+fileName_sul)
        img_oxido = sp.open_image(filePath+fileName_oxi)
    except Exception as err:
        print('Cannot open your training file.\nError info:' + str(err.args), end = '\n')
        exit(0)
    #input testing data
    input_testing_data()
    try :
        img_testing = sp.open_image(filePath + fileName_testing)
    except Exception as err:
        print('Cannot open your testing file.\n Error info:' + str(err.args), end = '\n')
        exit(0)
    # tranversing the img and cal spectral angle between testImg and refImg. 
    #Input: testing img and reference img.

    if 'Sulf' in fileName_testing: 
        res, accurarcy = Tranversing(img_oxido,img_sulfuro, img_testing, 2)
    else:
        res, accurarcy = Tranversing(img_oxido,img_sulfuro, img_testing, 1)

    width, height, deepth = img_testing.shape
    

    resName = fileName_testing.split('/')[1].split('_')[0] + '_res.bmp'
    show_res(res,accurarcy, width, height, filePath, resName)


    
    
    