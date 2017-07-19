import proxy 
import test_algorithm as ta
import os

# compute para table of the spectrum and return it. input a spectrum?
def Gaussian( spectrum):
	pass

# reference is standard specturm in the lib. load these sp of all minerals to be tested.
#attention, debugging here
def load_reference(filePath):
	file = open(filePath , 'r')
	lines = [line for line in file
	sp_dict = lines 
	return sp_dict

#attention, debugging here	
def load_amount(filePath):
	file = open(filePath, 'r')
	lines = [line for line in file]
	minerals_amount = lines
	
	return minerals_amount
	
# output proxy values to a .txt file.
def output_proxy(proxy_mineral, image_name):
	if 'proxy_mineral.txt' not in os.listdir('data/'):
		file_out = open('data/proxy_mineral.txt','w')
		file_out.write('\t\t')
		for key in sorted(proxy_mineral.keys()):
			file_out.write('\t%s\t' % key)
		file_out.write('\n')
		file_out.write('%s\t' % image_name)
		for key in sorted(proxy_mineral.keys()):
			file_out.write('\t%f\t' % proxy_mineral[key])
		file_out.write('\n')
		file_out.close()
	else:
		file_out = open('data/proxy_mineral.txt','a')
		file_out.write('%s\t' % image_name)
		for key in sorted(proxy_mineral.keys()):
			file_out.write('\t%f\t' % proxy_mineral[key])
		file_out.write('\n')
		file_out.close()
		
def cal_proxy_paraTable(fileName_image = 'unKnown.hdr', fileName_ref = 'unKnow2.hdr'):	
	#1. Read the pic file and got the sp of this file.
	filePath = 'data/'
	fileName_image = 'unKnown.hdr'
	fileName_ref = 'unKnow2.hdr'
	image_testing = ta.load_image(filePath + fileName )
	width, height, deepth = image_testing.shape
	sp_reference = load_reference(filePath + fileName_ref)

	#assign space for proxy_value of all the minerals. 
	proxy_mineral = {}
	for key in sp_reference.keys():
		proxy_mineral.setdefault(key,0.0)

	para_table_ref = {}	
	#2. got the reference parameters table. this is where proxy_value calculated differently 
	for key in sp_reference.keys():
		para_table_ref.setdefault(key, Gaussian(sp_reference))

	count_bg = 0
	for i in range(width):
		for j in range(height):
			sp_pixel = image_testing[i,j]
			
			if ta.exclude_BG:
				count_bg += 1
				continue
			# 2. got the testing pixels' para table.
			para_table_test = Gaussian(sp_pixel)
			
			# 3. Match the spectrum of reference and got sim of this two specturm( from table)
			
			for key in sp_reference: 
				sim = match_sim(para_table_ref[key], para_table_test)
				# 4. us ethe sim and give a percent to proxy
				proxy_mineral[key] += sim
	#5. cal the proxy value, this value is average proxy of the pixel, if all pixel is 100% mineralA so this rock is 100% mineralA
	for key in proxy_mineral:
		proxy_mineral[key] /= (width*height - count_bg)

	output_proxy(proxy_mineral, fileName_image)
	return proxy_mineral

#check_all could compute proxy values of all images, including minerals in 'file_Name_ref'. output the res to proxy_mineral autoly
def check_all():
	name_images = [name for name in os.listdir('data/REE/') if name.endswith('.hdr')]
	for name in name_images:
		cal_proxy_paraTable(fileName_image = name, fileName_ref = 'Unknow2.hdf')

#read the output file of check_all(), then draw scatter plot of minerals' amount and minerals' proxy value		
def plot(minerals_amount, proxy_file):
	#attention, debugging here.
	pass
	
	
def main():
	check_all()
	filePath = 'data/'
	output_name = 'proxy_mineral.txt'
	fileName_amount = 'minerals_amount.txt'
	minerals_amount = load_amount(filePath + fileName_amount)
	plot(minerals_amount, output_name)

if __name__ == '__main__':
	main()



	