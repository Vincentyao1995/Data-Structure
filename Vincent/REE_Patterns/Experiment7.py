import test_algorithm as ta
import spectral.io.envi as envi
import os
import ModifiedGaussianModel as MGM
import numpy as np
import matplotlib.pyplot as plt 
from scipy import optimize
import pre_processing_mineral as ppm
from scipy import signal

switch_PosCheck = 0
switch_test = 0
switch_smooth = 1
switch_bandTesting = 1
switch_multiChecking = 0

depth_threshold = 0.0075
method_possibility = 'general' # or general(none)
method_similarity = 'Original'



#this function input the absorption band index of a mineral (bastnas or Sth.), return the spectrum of reference band. 
def get_oringinal_spectrum(params_reference, band):
    filePath = 'data/'
    fileName = 'SpectraForAbsorptionFitting.hdr'
    sp_lib = envi.open(filePath + fileName)
    wavelength = sp_lib.bands.centers
    reflectance = sp_lib.spectra[0]
    spectrum = np.array([wavelength, reflectance]).T
    spectrum_band = choose_band(spectrum, params_reference, band)
    return spectrum_band

# this function is to adjust axis_y and axis_original_y to the same length, because they has different spectrum resolution. Input two spectrum([(920,0.222),....], [(919.08, 0.458),....] ) Resample and return two list:(reflectance):  (0.02,..) (0.5...)
def resample_sp_resolution(spectrum_band, spectrum_band_ori):

    if len(spectrum_band[:,0]) >= len(spectrum_band_ori[:,0]):
        axis_y = spectrum_band[:,1]
        axis_y_ori = signal.resample(spectrum_band_ori[:,1], len(spectrum_band[:,0]))
    else:
        axis_y_ori = spectrum_band_ori[:,1]
        axis_y = signal.resample(spectrum_band[:,1], len(spectrum_band_ori[:,0]))

    return axis_y, axis_y_ori


# compute para table of the spectrum and return it. input a spectrum?
def Gaussian(spectrum, params_reference, plotFileName = None):
    # divide the whole spectrum into 4 band, extract 4 ABP band in VNIR, and output these 4 bands' Gaussian params. 
    # params_reference provide the band info about start, end in the spectrum....
    params_testing = {}
    for band in sorted(params_reference.keys()):
        params_testing.setdefault(band)
        index_end = -1 
        index_begin = -1
        params_init = params_reference[band]['params_initial']
        
        spectrum_band = choose_band(spectrum, params_reference, band)
        
        axis_x = list(spectrum_band[:,0])
        axis_y = list(spectrum_band[:,1])
        params_testing[band] = MGM.fitting_leastSquare(spectrum_band, params_init)
        axis_y_fitting = MGM.multi_MGM(spectrum_band[:,0],list(params_testing[band]))
        index_begin = -1
        index_end = -1
        if plotFileName != None:
            plt.plot(axis_x, axis_y, lw = 2, c = 'green',label = 'oringinal curve')
            #plt.plot(axis_x, axis_y_smooth, lw = 1.5, c = 'yellow', label = 'smooth curve')
            plt.plot(axis_x,axis_y_fitting, lw = 1, c = 'red', label = 'fitting curve')
            fileName = plotFileName
            plt.legend()
            fileName += (str(band) + '.png')
            plt.savefig('data/fitting_res_REE/' + fileName)
            plt.close()
    return params_testing

# input a filePath and open this file, then read txt file info dict. return a dict containing initial params, optimize paras, in all bands.
def load_reference(filePath, type = 'normal'):
    if type != 'prewrittenFile':
        file = open(filePath , 'r')
        lines = [line for line in file]
    
        params_ref_dict = {}
        params_initial_mark_temp = 0
        params_optimize_mark_temp = 0
        band_index = ''
        for line in lines:
            if 'band' in line:
                band_index = line.split(':')[0]
                params_ref_dict.setdefault(band_index, {})
                (begin, end) = (line.replace('nm','').split()[1],line.replace('nm','').split()[-1])
                params_ref_dict[band_index].setdefault('begin', float(begin))
                params_ref_dict[band_index].setdefault('end', float(end))
                continue
            if 'initial' in line:
                params_initial_mark_temp = 1
                continue
            if 'optimize' in line:
                params_optimize_mark_temp = 1
                continue
            if params_initial_mark_temp == 1:
                params_initial_list = list(line.split())
                params_initial_list = [float(i) for i in params_initial_list]
                params_ref_dict[band_index].setdefault('params_initial', params_initial_list)
                params_initial_mark_temp = 0
                continue
            if params_optimize_mark_temp == 1:
                params_optimize_list = list(line.split())
                params_optimize_list = [float(i) for i in params_optimize_list]
                params_ref_dict[band_index].setdefault('params_optimize', params_optimize_list)
                params_optimize_mark_temp = 0
                continue
            if 'RMS' in line:
                params_ref_dict[band_index].setdefault('RMS', line)
            #time to append  init and optimize paras and RMS
        return params_ref_dict
    elif type == 'prewrittenFile':
        pass


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

# match similarity
def match_sim(params_ref, params_testing):

    params_ref_temp = []
    params_testing_temp = []
    for key in sorted(params_ref.keys()):
        params_ref_temp.extend(list(params_ref[key]['params_optimize']))
        params_testing_temp.extend(list(params_testing[key])) 
    sim = np.corrcoef(params_ref_temp, params_testing_temp)
    return sim

# this function cal the similarity between two Gaussian para lists. Input the image file and sum all pixels' sim and got a proxy. return this proxy value.
def cal_proxy_paraTable(fileName_image = 'unKnown.hdr', fileName_ref = 'unKnow2.hdr'):
    #1. Read the pic file and got the sp of this file. Also read params of standard curve.
    filePath = 'data/VNIR/rocks/'
    fileName_image = 'VNIR_sample1_18points.hdr'
    fileName_ref = 'bastnas_gau_params.txt'
    image_testing = ta.load_image(filePath = filePath + fileName_image )
    width, height, deepth = image_testing.shape
    #2. Get the parameters table of reference spectrum
    params_reference = ppm.load_reference(filePath + fileName_ref)
    proxyValue = 0
    
    count_bg = 0
    for i in range(width):
        for j in range(height):
            print('pixel %d, %d processing, total: %d\n' % (i,j, width*height))
            ref_pixel = image_testing[i,j]
            band_pixel = image_testing.bands.centers
            sp_pixel = np.array([band_pixel, ref_pixel]).T

            if ta.exclude_BG(ref_pixel):
                count_bg += 1
                continue
            # 2. got the testing pixels' para table.  #2,4 2,5 2,6 is three ree pixel. 3,1 is noise pixel
            if i == 5 and j == 1:
                fileName_output = 'pixel_x2y6'
                params_testing = Gaussian(sp_pixel, params_reference,plotFileName = fileName_output)
            elif i == 0 and j == 2:
                fileName_output = 'pixel_x3y1'
                params_testing = Gaussian(sp_pixel, params_reference,plotFileName = fileName_output)
            else:
                params_testing = Gaussian(sp_pixel, params_reference)
            
            # 3. Match the spectrum of reference and got sim of this two specturm( from table). Attention: didn't match multiple minerals yet, only get one. So 4.5. is changed from the initial code.(Exp7)
            sim = match_sim(params_reference, params_testing)

            # 4. use the sim and give a percent to proxy
            #proxy_mineral[key] += sim
            proxyValue += sim
    #5. cal the proxy value, this value is average proxy of the pixel, if all pixel is 100% mineralA so this rock is 100% mineralA
    #for key in proxy_mineral:
    #    proxy_mineral[key] /= (width*height - count_bg)

    #output_proxy(proxyValue, fileName_image)
    return proxyValue



# input the whole spectrum, and params_reference(a dict) read from reference Gaussian parameters .txt file and the band u want to choose, return the band's spectrum
def choose_band(spectrum, params_reference, band):

    index_end = -1 
    index_begin = -1
        
    #find the spectrum band in pixels' specturm, because reference sp has different sp resolution with testing sp.
    for i in range(len(spectrum[:,0])):
        if spectrum[:,0][i] <= params_reference[band]['begin'] and spectrum[:,0][i+1] >= params_reference[band]['begin']:
            index_begin = i
        elif spectrum[:,0][i] <= params_reference[band]['end'] and spectrum[:,0][i+1] >= params_reference[band]['end']:
            index_end = i
    if index_begin != -1 and index_end != -1:
        axis_x = spectrum[:,0][index_begin:index_end]
        axis_y = spectrum[:,1][index_begin:index_end]
        
        spectrum_band = np.array([axis_x, axis_y]).T
        return spectrum_band

#check_all() check all the images, and all the pixels in them. Output every pixels' proxy value(Maybe scaling, or similarity computed using other alg.) into txt files
def check_all(initial_weight = 1.0):
    filePath = 'data/VNIR/ASCII_VNIR/'
    if switch_test ==1:
        filePath = 'data/VNIR/' 
    filePath_image_temp = 'data/VNIR/rocks/VNIR_sample1_18points.hdr'
    wavelength_pixel = ta.load_image(filePath = filePath_image_temp).bands.centers
    name_images = [name for name in os.listdir(filePath) if name.endswith('.txt')]
    params_reference = load_reference('data/VNIR/rocks/' + 'bastnas_gau_params.txt')#attention, different mineral, rewrite this function? And this become a pre-txt file. Read info from txt.
    params_reference.pop('band4')

    filePath_output = 'output/VNIR_scaling/'
    if switch_multiChecking != 1:
        fileName_output_picScaling = filePath_output + 'pics_scaling_bands.txt'
        fileName_output_picScaling = filePath_output + 'multiChecking.txt'
        file_output_picScaling = open(fileName_output_picScaling,'w')
    
    dict_proxyValue = {'band1':{initial_weight: []}, 'band2':{initial_weight: []},'band3':{initial_weight: []}}

    #image loop, process all images.
    for name in sorted(name_images):
        print('image %s processing!' % name)
        # open the txt file and got all pixels' spectrum
        image_file = open(filePath + name)
        lines = [line for line in image_file]

        if switch_multiChecking != 1:
            # open txt file to write every pixels' scaling
            if switch_PosCheck == 1:
                fileName_output = filePath_output + 'Possibility_' + name.split('_')[-1]
            elif switch_smooth and method_similarity == 'Gaussian':
                fileName_output = filePath_output + 'Gaussian_Smooth_scaling_' + name.split('_')[-1]
            else:
                fileName_output = filePath_output + method_similarity +'Original_Smooth_scaling_' + name.split('_')[-1]
            file_output = open(fileName_output , 'w')

            file_output.write('Gaussian_Smooth_scaling band 1-4(Bastnas)\n')

            file_output_picScaling.write('\t\t band1 \t band2 \t band3 \t band4\n')
        
        scaling_temp = {'band1': 0.0 , 'band2': 0.0,'band3': 0.0,'band4': 0.0}
        scaling_pic = 0. 
        count_pixel_num = len(lines) - 8
        
        #pixel loop, process all pixels in image.(through a ROI ASCII file)
        for line_index in range(len(lines)):
            print('sample%d processing: %f\n' % ( name_images.index(name)+1 , line_index/len(lines) ))
            
            line = lines[line_index]
            # ignore the header info in image(.txt file)
            if line_index <= 7 or len(line.split()) < 100:
                continue
            #got useful info(pixels' spectrum) from txt. Pixels' coordinate and its reflectance and wavelength
            ID = line.split()[0] # ID, x, y ,spectrum
            (x, y) = (int(line.split()[1]), int(line.split()[2]))
            ref_pixel = list(line.split()[3:])
            ref_pixel = [float(i) for i in ref_pixel]
            sp_pixel = np.array([wavelength_pixel, ref_pixel]).T
            scaling_pixel = {}
            
            #band loop, compute the scaling in different band separately.
            for band in sorted(params_reference.keys()):

                if switch_bandTesting:
                    if band == 'band1' or band == 'band3':
                        continue

                spectrum_band = choose_band(sp_pixel, params_reference, band)#attention, choose band according to different absorption position of different mineral. Maybe make a .txt file including all info.
                axis_x, axis_y = list(spectrum_band[:,0]),list(spectrum_band[:,1])

                if switch_smooth == 1:
                    axis_y = ppm.savitzky_golay(axis_y,7,3)
                    spectrum_band = np.array([axis_x,axis_y]).T

                #cal the sim between reference and sp_pixel.
                reference_info = params_reference[band]
                possibility = ppm.cal_possibility(reference_info, spectrum_band, depth_threshold = depth_threshold, method = method_possibility)#attention, different mineral
                if possibility == 0.0: #attention, ouput sim here. 
                    scaling = 0.0
                    similarity = 0.0

                elif band == 'band2' and possibility <0.9:
                    scaling = 0.0
                    similarity =0.0
                    
                else:
                    #only output possibility.
                    if switch_PosCheck == 1:
                        scaling = float(possibility)
                        scaling_pixel.setdefault(band,scaling)
                        scaling_temp[band] += scaling
                        continue

                    #None- smooth Gaussian scaling match: 
                    if method_similarity == 'Gaussian':
                        similarity = ppm.cal_similarity(MGM.multi_MGM(axis_x, list(params_reference[band]['params_optimize'])), axis_y, method = method_similarity)
                    else:
                        # get the original spectrum of reference spectrum
                        spectrum_band_ori = get_oringinal_spectrum(params_reference,band)#attention, different minerals
                        
                        # make sure and adjust axis_y and axis_original_y has the same length, because they has different spectrum resolution
                        axis_y, axis_y_ori = resample_sp_resolution(spectrum_band, spectrum_band_ori)
                        
                        # frechet and hausdorff require at least two dimension data. So add axis_x into it.
                        if method_similarity == 'Frechet' or method_similarity == 'Hausdorff' or method_similarity == 'Procrustes':
                            axis_y_ori = np.array([axis_x, axis_y_ori]).T
                            axis_y = np.array([axis_x, axis_y]).T

                        
                        # got the similarity between reference sp and testing sp
                        similarity = ppm.cal_similarity(axis_y_ori,axis_y, method = method_similarity)
                        
                    scaling = float(similarity *possibility)#attention, output similarity. sim * pos here

                
                scaling_pixel.setdefault(band,scaling)
                scaling_temp[band] += scaling

            if switch_multiChecking != 1: 
                #write the result. ouput scaling of all pixels into one file. Then use excel to do analysis work
                file_output.write('%d \t %d \t ' % (x,y))
                for band in sorted(scaling_pixel.keys()):
                    file_output.write('%f\t' % scaling_pixel[band])
                file_output.write('\n')
            
            scaling_pic += ppm.cal_scaling(scaling_pixel)
            #end of checking one picture's all lines(pixels)
        
        #write proxy info of one picture's all bands. 
        if switch_multiChecking != 1:
            file_output.write('depth threshold: %f\n' % depth_threshold)
            file_output.close()

            #write the pic's scaling to 'picScaling.txt'
            file_output_picScaling.write(name + '\t')
            for band in sorted(scaling_temp.keys()):
                file_output_picScaling.write('%f\t' % float(scaling_temp[band]/ count_pixel_num))
            file_output_picScaling.write('\nSummation of Scaling: %f \t %f \t %f \t %f, the number of total pixels: %d\n' % (scaling_temp['band1'],scaling_temp['band2'],scaling_temp['band3'],scaling_temp['band4'], count_pixel_num))
            file_output_picScaling.write('Picture total scaling: %f average: %f \n \n '% (scaling_pic, float(scaling_pic/count_pixel_num) ) )

        #save proxy info of one picture's all bands.     
        elif switch_multiChecking == 1:
            for band in sorted(scaling_temp.keys()):
                dict_proxyValue[band][initial_weight].append(scaling_temp[band]/ count_pixel_num)
            #attention, time to debug writing check multiple proxy value
        #end of checking all pictures.
    if switch_multiChecking != 1:        
        file_output_picScaling.write('center match (possibility) method: %s\t scaling(similarity) method: %s \t Smooth or not: %d \n' % (method_possibility, method_similarity, switch_smooth))
        file_output_picScaling.close()
    return dict_proxyValue
    # end of check_all()
def main():
    check_all()
    proxyValue = cal_proxy_paraTable(fileName_image = 'VNIR_sample1_18points.hdr',fileName_ref = 'bastnas_gau_params.txt')
    print(proxyValue)
    quit()
    filePath = 'data/'
    output_name = 'proxy_mineral.txt'
    fileName_amount = 'minerals_amount.txt'
    #minerals_amount = load_amount(filePath + fileName_amount)


if __name__ == '__main__':
    check_all(initial_weight = 0.8)



    
