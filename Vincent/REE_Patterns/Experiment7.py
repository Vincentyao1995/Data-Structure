import test_algorithm as ta
import os
import ModifiedGaussianModel as MGM
import numpy as np
import matplotlib.pyplot as plt 
from scipy import optimize
import pre_processing_mineral as ppm

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

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
        #axis_y_smooth = savitzky_golay(spectrum_band[:,1], 11, 3) # window size 51, polynomial order 3
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

# reference is standard specturm in the lib. load these sp of all minerals to be tested.
def load_reference(filePath):
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
    params_reference = load_reference(filePath + fileName_ref)
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

# this function compute scaling, input listA and listB, and return the scaling(0-1) (s*listA = listB) A is fitting list of standard spectrum library
def cal_scaling(listA, listB):
    scaling = 0.
    errFunc = lambda s, x,y: (y- s* x)**2
    scaling, success = optimize.leastsq(errFunc, scaling, args=(listA,listB), maxfev = 20000) 
    return scaling

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
        #axis_y_smooth = savitzky_golay(axis_y, 11, 3) # window size 51, polynomial order 3
        spectrum_band = np.array([axis_x, axis_y]).T
        return spectrum_band

#check_all() check all the images, and all the pixels in them. Output every pixels' proxy value(Maybe scaling, or similarity computed using other alg.) into txt files
def check_all():
    filePath = 'data/VNIR/ASCII_VNIR/'
    filePath = 'data/VNIR/'
    filePath_image_temp = 'data/VNIR/rocks/VNIR_sample1_18points.hdr'
    wavelength_pixel = ta.load_image(filePath = filePath_image_temp).bands.centers
    name_images = [name for name in os.listdir(filePath) if name.endswith('.txt')]
    params_reference = load_reference('data/VNIR/rocks/' + 'bastnas_gau_params.txt')

    #image loop, process all images.
    for name in sorted(name_images):
        print('image %s processing!' % name)
        # open the txt file and got all pixels' spectrum
        image_file = open(filePath + name)
        lines = [line for line in image_file]
        # got the pixels' spectrum, ignore the header of file.
        fileName_output = 'Originalnon-smoothScaling_' + name.split('_')[-1]
        file_output = open(fileName_output , 'w')
        file_output.write('Original non-smooth Scaling band 1-4(Bastnas)\n')
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
                spectrum_band = choose_band(sp_pixel, params_reference, band)
                axis_x, axis_y = list(spectrum_band[:,0]),list(spectrum_band[:,1])
                
                #cal the sim between reference and sp_pixel.
                reference_info = params_reference[band]
                reference_info = list(reference_info)
                similarity = ppm.cal_similarity(reference_info, sp_pixel)

                #None- smooth Gaussian scaling match: 
                scaling = cal_scaling(MGM.multi_MGM(axis_x, list(params_reference[band]['params_optimize'])), axis_y)
                #scaling = cal_scaling(axis_y, ori_lib_spectrum_band)
                scaling_pixel.setdefault(band,scaling)

            #write the result. ouput scaling of all pixels into one file. Then use excel to do analysis work
            file_output.write('%d \t %d \t ' % (x,y))
            for band in scaling_pixel:
                file_output.write('%f\t' % scaling_pixel[band])
            file_output.write('\n')

        file_output.close()

    
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
    check_all()



    