import test_algorithm as ta
import Classifier_SAM as SAM
from PIL import Image 
import pandas as pd
import numpy as np

pi = 3.1415926

# Switch, cal the pixel's angle and see it's range or not. 0 is default, better recommend. cal the avg sp in different band
all_pixel_sp = 0

def get_range(ori_angle,num = 16):
    for i in range(num):
        if ori_angle * 100 >= i and ori_angle * 100 < i+1:
            return i 

#1. Open an image # sulf01 04 49 50.
filePath = 'data/sulfuros/'
fileName = 'EscSulf50_Backside_SWIR_Subset_Masked.hdr'
image = ta.load_image(filePath = filePath + fileName)

#2. Cal the average spectrum of this img.
sp_average = SAM.cal_avg_SP(image)

#3. Tranverse the whole pic, input every pixel and avg sp to cal spectrum angle.
width, height, deepth = image.shape
step = int ((2530.15-928.08)/ deepth)
wavelength = [i* step for i in range(deepth) ]

image_new = Image.new('RGB', (width, height),color = (0,0,0))

file_out = open (fileName + '_sp_angle.txt','w')
image_out = fileName.split('_')[0] + '_sp_angle.jpg'
file_out.write('pixel_index\t\tangle\t\twavelength\n')
pixel_index = 0

#about 7. get the avg sp in each range. sulf01:0.01-0.17 : 16 ; sulf04:0.00-0.15 15
sp_range = []
count_range = []
avg_angle_range = []
for i in range(15):
    count_range.append(0)
    sp_range.append([])
    avg_angle_range.append(0)

for i in range(width):
    for j in range(height):
        sp_pixel = image[i,j]
        # skip background pixel
        if SAM.exclude_BG(sp_pixel):
            pixel_index += 1
            print('processing: %f! \n' % float(pixel_index/width/height) )
            continue
        ori_angle = SAM.cal_sp_angle(sp_pixel, sp_average)
        
        if all_pixel_sp == 1:
            # 4. color the new pic. 
            
            #nomalize the angle.
            angle = ori_angle/pi
            color = angle *255
            image_new.putpixel((i,j),(int(color+0.5),0,0)) #attention, rgb white - red - black
            
            # 5. save pixels' angle and spectrum into dataframe and then write them.
            file_out.write('%d \t\t %.6f \t\t' % (pixel_index, ori_angle) + '\n')
            #file_out.write('%d \t\t %.6f \t\t' % (pixel_index, ori_angle) + ''.join(str(sp_pixel).replace('\n', '')) + '\n')
            pixel_index += 1
            print('processing: %f! \n' % float(pixel_index/width/height) )
        #7. later, I found that angles' range is '0.01-0.16', but pixels are so many, then we split them into 16 gourps and get the average sp of each group.
        else:
            range_index = get_range(ori_angle)
            
            if sp_range[range_index] == []:
                sp_range[range_index] += list(sp_pixel)
                count_range[range_index] += 1
                avg_angle_range[range_index] += ori_angle
                continue
                
            for d in range(len(sp_pixel)):
                sp_range[range_index][d] += sp_pixel[d]
            # cal the number of pixels in each band, and its sum angle.	
            count_range[range_index] += 1
            avg_angle_range[range_index] += ori_angle
            pixel_index += 1
            print('processing: %f! \n' % float(pixel_index/width/height) )
if all_pixel_sp == 0:
    #cal the avg angle, sp.
    assert len(sp_range) == len(count_range) == len(avg_angle_range), 'check code, sth wrong.'
    file_out.write('average sp of pic: ')
    file_out.write(str(sp_average).replace('\n','') + '\n')
    for i in range(len(sp_range)):
        if avg_angle_range[i] == 0:
            continue
        avg_angle_range[i] /= count_range[i]
        for d in range(len(sp_range[8])):
            sp_range[i][d] /= count_range[i]

    for i in range(len(count_range)):
        file_out.write('%f - %f \t %f \t ' % ( (i+0.0)/100, (i+1.0)/100, avg_angle_range[i]))
        file_out.write(str(sp_range[i]) + '\n')
    print('done!')
    file_out.close()
#6. output the spectrum and angle.
else:
    image_new.save(filePath.split('/')[0] + image_out)
    file_out.close()
    print('done!\n')





