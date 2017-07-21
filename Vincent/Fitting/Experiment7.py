import proxy 
import test_algorithm as ta
import os
import ModifiedGaussianModel as MGM
import numpy as np
import matplotlib.pyplot as plt 

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
def Gaussian(spectrum, params_reference):
    # divide the whole spectrum into 4 band, extract 4 ABP band in VNIR, and output these 4 bands' Gaussian params. 
    # params_reference provide the band info about start, end in the spectrum....
    params_testing = {}
    for band in sorted(params_reference.keys()):
        params_testing.setdefault(band)
        index_end = -1 
        index_begin = -1
        params_init = params_reference[band]['params_initial']
        
        #find the spectrum band in pixels' specturm, because reference sp has different sp resolution with testing sp.
        for i in range(len(spectrum[:,0])):
            if spectrum[:,0][i] <= params_reference[band]['begin'] and spectrum[:,0][i+1] >= params_reference[band]['begin']:
                index_begin = i
            elif spectrum[:,0][i] <= params_reference[band]['end'] and spectrum[:,0][i+1] >= params_reference[band]['end']:
                index_end = i
        if index_begin != -1 and index_end != -1:
            axis_x = spectrum[:,0][index_begin:index_end]
            axis_y = spectrum[:,1][index_begin:index_end]
            axis_y_smooth = savitzky_golay(axis_y, 51, 3) # window size 51, polynomial order 3
            plt.plot(axis_x, axis_y)
            plt.plot(axis_x, axis_y_smooth)
            spectrum_band = np.array([axis_x, axis_y_smoooth]).T
            params_testing[band] = MGM.fitting_leastSquare(spectrum_band, params_init)
            MGM.plot_figures(params_testing[band],spectrum[:,0][index_begin:index_end], spectrum[:,1][index_begin:index_end])
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
            ref_pixel = image_testing[i,j]
            band_pixel = image_testing.bands.centers
            sp_pixel = np.array([band_pixel, ref_pixel]).T

            if ta.exclude_BG(ref_pixel):
                count_bg += 1
                continue
            # 2. got the testing pixels' para table. 
            params_testing = Gaussian(sp_pixel, params_reference)
            
            # 3. Match the spectrum of reference and got sim of this two specturm( from table). Attention: didn't match multiple minerals yet, only get one. So 4.5. is changed from the initial code.(Exp7)
            sim = match_sim(params_reference, params_testing)
            #attention, code here, time to output params, and see the res of pixel sp' fitting. And find the scaling.
            # 4. use the sim and give a percent to proxy
            #proxy_mineral[key] += sim
            proxyValue += sim
    #5. cal the proxy value, this value is average proxy of the pixel, if all pixel is 100% mineralA so this rock is 100% mineralA
    #for key in proxy_mineral:
    #    proxy_mineral[key] /= (width*height - count_bg)

    #output_proxy(proxyValue, fileName_image)
    return proxyValue

#check_all could compute proxy values of all images, including minerals in 'file_Name_ref'. output the res to proxy_mineral autoly
def check_all():
    name_images = [name for name in os.listdir('data/VNIR/rocks/') if name.endswith('.hdr')]
    for name in name_images:
        cal_proxy_paraTable(fileName_image = name, fileName_ref = 'Unknow2.hdf')

#read the output file of check_all(), then draw scatter plot of minerals' amount and minerals' proxy value		
def plot(minerals_amount, proxy_file):
    #attention, debugging here.
    pass
    
    
def main():
    #check_all()
    cal_proxy_paraTable(fileName_image = 'VNIR_sample1_18points.hdr',fileName_ref = 'bastnas_gau_params.txt')
    filePath = 'data/'
    output_name = 'proxy_mineral.txt'
    fileName_amount = 'minerals_amount.txt'
    minerals_amount = load_amount(filePath + fileName_amount)
    plot(minerals_amount, output_name)

if __name__ == '__main__':
    main()



    