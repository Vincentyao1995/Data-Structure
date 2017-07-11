import test_algorithm as ta
import Classifier_SAM as SAM
from PIL import Image 
import pandas as pd
import numpy as np

pi = 3.1415926

#1. Open an image # sulf01 04 49 50.
filePath = 'data/sulfuros/'
fileName = 'EscSulf01_Backside_SWIR_Subset_Masked.hdr'
image = ta.load_image(filePath = )

#2. Cal the average spectrum of this img.
sp_average = SAM.cal_avg_SP(image)

#3. Tranverse the whole pic, input every pixel and avg sp to cal spectrum angle.
width, height, deepth = image.shape
df_sp_angle = pd.DataFrame(np.zeros((width*height, deepth+1)), columns = wavelength)
#attention, time to fill wavelength and then save pixel angle and sp into dataFrame.

for i in range(width):
	for j in range(height):
		sp_pixel = image[i,j]
		# skip background pixel
		if SAM.exclue_BG(sp_pixel):
			continue
		angle = SAM.cal_sp_angle(sp_pixel, sp_average)
		# 4. color the new pic. 
		
		#nomalize the angle.
		angle = angle/pi
		color = angle *255
		image_new[i,j] = color #attention, rgb black - red - white
		
		

