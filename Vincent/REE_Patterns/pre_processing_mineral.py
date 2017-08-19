from scipy import signal
from scipy import optimize
from numpy import mean
import ModifiedGaussianModel as MGM
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean
import frechet
import math

center_error = 6
threshold_center_error = 3

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


# function of smooth    
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

#this function input the spectrum(rocks pixels'), centers' position and weight([863.5,80%]), and the judging method('general', 'modeling'). Return the percent this pixel is possible to be mineralA
def cal_centers_around(sp_testing, center, method = 'general',switch_minima_centerMatching = 1):
    
    if method == 'general':
        for i in range(len(sp_testing[:,0])):
            if sp_testing[:,0][i] <= center[0] and sp_testing[:,0][i+1] >= center[0]: 
                index = i
        left_higher_num = 0
        right_higher_num = 0
        # set acceptable error
        index_list = [index-2, index -1, index, index+1, index+2 ]
        res = False
        if switch_minima_centerMatching != 1:
            for index in sorted(index_list):
                for i in range(index-3, index+4):
                    if i < index:
                        if sp_testing[:,1][i] >= center[1]:
                            left_higher_num += 1
                    if i > index:
                        if sp_testing[:,1][i] >= center[1]:
                            right_higher_num += 1

                if (left_higher_num >= 1 and right_higher_num >= 2) or (left_higher_num >= 2 and right_higher_num >= 1):
                    res = True
                    break
                else:
                    continue
        elif switch_minima_centerMatching == 1:
            minima_index = signal.argrelextrema(savitzky_golay(sp_testing[:,1], 5,3), np.less)
            for i in sorted(list(minima_index[0])):
                minima_position = sp_testing[:,0][i]
                if center[0] >= minima_position - threshold_center_error and center[0] <= minima_position + threshold_center_error:
                    res = True
                    break
        return res

    if method == 'modeling':
    # use Gaussian model to fit and got center.
        params_initial = []
        height = [0.01 for i in range(len(center))]
        width = [5. for i in range(len(center))]
        params_initial.extend(height)
        params_initial.extend(width)
        params_initial.extend(center)
        params_initial.extend([0])

        params_optimize = MGM.fitting_leastSquare(sp_testing, params_initial)

        num_params_group = int(len(params_optimize)/3)
        centers_modeling = params_optimize[-num_params_group - 1 : -1]
        res_list = []
        for i in range(len(center)):
            if centers_modeling[i] >= center[i] - center_error and centers_modeling[i] <= center[i] + center_error:
                res_list.append(1)
            else:
                res_list.append(0)
        return res_list

#this function input centers position, return a dict, key is centers' position and value is weight. [720.056: 0.3, 760.58: 0.7]
def cal_centers_weight(centers_position, mineral_type = 'bastnas', initial_weight = 0.9):
    centers_weight = {}

    #in this for loop, u could read different mineral centers info from a txt file. attention, First write a file contains centers and weight after read this file, u could mienralX - weights in bands. 
    for center in sorted(centers_position):
        if mineral_type == 'bastnas':
            if int (sum(centers_position)/ len(centers_position)) in range(705,770):
                if center == 740:
                    centers_weight.setdefault(center, initial_weight)
                else:
                    centers_weight.setdefault(center, float((1.0-initial_weight)/5))
            if int (sum(centers_position)/ len(centers_position)) in range(770,833):
                if center == 791 or center == 797:
                    centers_weight.setdefault(center, initial_weight/2.0)
                else:
                    centers_weight.setdefault(center, (1.0-initial_weight/2.0)/4)
            if int (sum(centers_position)/ len(centers_position)) in range(854,880):
                if center == 863 :
                    centers_weight.setdefault(center, initial_weight)
                else:
                    centers_weight.setdefault(center, (1.0-initial_weight)/2)
            if int (sum(centers_position)/ len(centers_position)) in range(880,900):
                if center == 880 :
                    centers_weight.setdefault(center, 1.0)
                else:
                    centers_weight.setdefault(center, 0.0)
    return centers_weight

# input the reference spectrum info(a list), including this mineral spectrum's main feature, like centers position and depth. And testing spectrum need to be tested. return the simlarity(0-100%) between ref and testing. This function is to make sure whether this spectrum is possible to be mineralA (reference mineral spectrum)
def cal_possibility(reference_info, sp_testing, depth_threshold = 0.0075, method = 'general', weight = 0.8):
    #initial part and scoring system: only use center to score and evaluate similarity.
    
    num_param_group = int(len(reference_info['params_initial'])/3)
    centers_position = reference_info['params_initial'][-num_param_group - 1 : -1]
    #there, users should input the weigth of each center: double ABP: double abp center 90%, other centers occupies 10%; single abp center 80% other centers shares 20%.
    centers = cal_centers_weight(centers_position, initial_weight = weight)

    
    index_minimum = sp_testing[:,1].argmin()
    index_mark = 0
    if index_minimum == 0:
        index_mark = 1
    absorption_depth = (mean(sp_testing[:,1][0:3]) + mean(sp_testing[:,1][-4:-1]))/2 - mean(sp_testing[:,1][index_minimum-1:index_minimum+2])
    
    sim_percent = 0.
    if absorption_depth < depth_threshold or index_mark == 1:
        sim_percent = 0.
        return sim_percent

    else:
        if method == 'general':
            for center_position in sorted(centers.keys()):
                if cal_centers_around(sp_testing, [center_position, centers[center_position]], method = method):
                    sim_percent += centers[center_position]
        elif method == 'modeling':
            res_list = cal_centers_around(sp_testing, sorted(list(centers.keys())), method = method)
            for i in range(len(res_list)):
                if res_list[i] == 1:
                    center_position = sorted(centers.keys())[i]
                    sim_percent += centers[center_position]

    return sim_percent

# this function compute scaling, input listA and listB, and return the scaling(0-1) (s*listA = listB) A is fitting list of standard spectrum library
def cal_similarity(listA, listB, method = 'Gaussian'):
    if method =='Gaussian' or method == 'Original':
        scaling = 0.
        errFunc = lambda s, x,y: (y- s* x)**2
        scaling, success = optimize.leastsq(errFunc, scaling, args=(listA,listB), maxfev = 20000) 
        return scaling

    elif method == 'Corrcoef':
        return np.corrcoef(listA, listB)[0][1]
    elif method == 'Frechet':
        dist = frechet.frechetDist(listA, listB)
        return 1 / (1 + dist)
    elif method == 'DTW':
        from dtw import dtw
        dist = dtw(listA, listB, dist = lambda x,y : math.sqrt((x-y)**2))
        return 1 / (1 + dist)
    elif method == 'Hausdorff':

        if 0:
            params = [10.,2.5]
            errFunc = lambda params, x,y: (y- (params[0]* x - params[1]))**2
            params, success = optimize.leastsq(errFunc, params, args=(listA[:,1],listB[:,1]), maxfev = 20000) 
            plt.plot(listA[:,0], listA[:,1])
            listA[:,1] = listA[:,1]* params[0] - params[1] 
            plt.plot(listA[:,0], listA[:,1])
            plt.plot(listB[:,0], listB[:,1])
            plt.show()

        from scipy.spatial.distance import directed_hausdorff
        dist = max( directed_hausdorff(listA , listB)[0], directed_hausdorff(listB, listA)[0] )
        return 1/ (1 + dist)

    elif method == 'Procrustes':
        from scipy.spatial import procrustes
        matrixA, matrixB, dist = procrustes(listA, listB)
        return 1 / (1 + dist)
        
#this method receive two 2-d or 1-d lists, return the procrustes results of them. (Done similarity transformation)
def cal_procrustes(axis_y_ori, axis_y):
    pass
    from scipy.spatial import procrustes
    
        
# this method need pre-geologist knowledge, so cal this info auto-matically is kind of hard.
# The most similar: use MGM to simulate reference spectrum and use 'Gaussian params' as reference info.
def cal_reference_info(sp_reference):
    
    return reference_info

# input a pixel's scaling dict, band1: 0.856, band2: 0.76 ..... return a scaling value of this pixel, use different weight to compute.
def cal_scaling(scaling_dict):

    #record how many bands have none-zero scaling
    num_band_noneScaling = 0 
    scaling_dict_noneZero = {}
    for band in sorted(scaling_dict.keys()):
        if scaling_dict[band] != 0.0:
            num_band_noneScaling += 1
            scaling_dict_noneZero.setdefault(band,scaling_dict[band])
    if num_band_noneScaling == 0:
        return 0.
    elif num_band_noneScaling == 1:
        band = list(scaling_dict_noneZero.keys())[0]
        return 0.4 * mean(list(scaling_dict_noneZero.values()))
    elif num_band_noneScaling == 2:
        return 0.8 * mean(list(scaling_dict_noneZero.values()))
    elif num_band_noneScaling == 3:
        return mean(list(scaling_dict_noneZero.values()))
    else:
        return mean(list(scaling_dict_noneZero.values()))

# this function input a spectrum(band), return its ABP depth, alg is DT's 'depth - proxy value'. 
def cal_absorption_depth(spectrum):
    return depth

# input a filePath and open this file, then read txt file info dict. return a dict containing initial params, optimize paras, in all bands.
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





