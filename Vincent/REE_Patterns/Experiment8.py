import spectral.io.envi as envi

#this script is do Gaussian modeling for multiple minerals and output the paramters like DT's method. 

def open_spectraData(filePath):
	sp_lib = envi.oepn(filePath)

	return sp_lib
def get_initialParams(filePath, sp_name):
	file = open(filePath, 'r')
	lines = [line for line in file]
	dict_mineral_initParams = {}
	flag_mineralReading = 0
	
	for line in lines:
		if 'mineral' in line:
			mineralName = line.split('mineral').split(':')[0]
			dict_mineral_initParams.setdefault(mineralName, {})
			flag_mineralReading = 1
		if flag_mineralReading == 1:
			if 'band' in line:
				begin = line.split(':')[1].split('-')[0]
				end = line.split(':')[1].split('-')[1]
				dict_mineral_initParams[mineralName].setdefault('begin', begin)
				dict_mineral_initParams[mineralName].setdefault('end', end)
				
			elif 'height' in line:
				height = line.split(':')[1].split('\t')
				height = [float(h) for h in height]
				dict_mineral_initParams[mineralName].setdefault('height', height)
			elif 'width' in line:
				width  = line.split(':')[1].split('\t')
				width = [float(w) for w in width]
				dict_mineral_initParams[mineralName].setdefault('width', width)
			elif 'center' in line:
				center = line.split(':')[1].split('\t')
				center = [float(c) for c in center]
				dict_mineral_initParams[mineralName].setdefault('center', center)
			elif 'yshift' in line:
				yshift = line.split(':')[1]
				flag_mineralReading = 0
				dict_mineral_initParams[mineralName].setdefault('yshift', yshift)
				#attention, time to debugging, and next step is Gaussian Modeling.
			
	
if __name__ == '__main__':
	
	filePath = 'data/'
	fileName_lib = 'SpectraForAbsorptionFitting.hdr'
	fileName_initialParams = 'initialParams_Minerals.txt'
	sp_lib = open_spectraData(filePath + fileName_lib)
	
	wavelength = sp_lib.bands.centers
	
	
	
	for i in range(len(sp_lib.spectra)):
		reflectance = sp_lib.spectra[i]
		sp_name = sp_lib.spectra[i].name# attention, want to see the name of spectra so that I could judge from name to read .txt file.
		
		initial_paramters = get_initialParams(filePath + fileName_initialParams, sp_name)
			
			
			
			
	
	