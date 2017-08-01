import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import spectral.io.envi as envi
import tensorflow as tf
from scipy import optimize


switch_excel = 0
switch_envi = 1
switch_dataFrame = 0
switch_sp_choice_envi = 0
switch_mutiBands = 1

#Modified gaussian Model
def MGM(x,height, center, width, yshift, n = -1):
    if type(x) == tf.Tensor:
        return yshift + height * tf.exp( - (x**n - center**n)**2 / (width*2) )
    else:
        return yshift + height * np.exp( - (x**n - center**n)**2 / (width*2) )

#Multiple MGM, input a para list, then they would use the list to construct muti- Gaussian. list construction: [h1,h2,...,hx, c1,...,cx, w1,...,wx]
def multi_MGM(x, params, n = 1):
    assert len(params)%3 == 1, 'your input params has different number of width, height and center'
    res = 0
    height = params[:int(len(params)/3)]
    width = params[int(len(params)/3): 2*int(len(params)/3)]
    center = params[2*int(len(params)/3):-1]
    yshift = params[-1]
    num_Gaussian = int(len(params)/3)

    for i in range(num_Gaussian):
        newGaussian = MGM(np.array(x),height[i],center[i],width[i],yshift = yshift, n = n)
        res += newGaussian
    return res

#original gaussian function
def gaussian(x, params ):
    assert len(params) == 4, 'your input params has different number of width, height and center'
    res = 0
    height = params[0]
    width = params[1]
    center = params[2]
    yshift = params[3]
    return yshift + height * np.exp( - (x - center)**2 / (width*2) )

# Convert 30 dimension reflectance to 256D. Methods: interpolate the hull' function using around 30 points, then cal all wavelength points(256). input an dataFrame hull, whose columns = ['wavelength','reflectance'] and the wavelength list of all 256 dimension. return 256 dimension's reflectance. 
def resample(hull, wavelengths):
    f = interp1d(hull['wavelength'], hull['reflectance'])
    return f(wavelengths)

# core alg. this funtion input an list, [[928,0.25], [935,0.41]....] return the continuum(hull) of this specturm(curve)
def qhull(sample):

    link = lambda a,b: np.concatenate((a,b[1:]))
    edge = lambda a,b: np.concatenate(([a],[b]))

    def dome(sample,base): #alg
        h, t = base
        dists = np.dot(sample-h, np.dot(((0,-1),(1,0)),(t-h)))
        outer = np.repeat(sample, dists>0, axis=0)
        
        if len(outer):
            pivot = sample[np.argmax(dists)]
            return link(dome(outer, edge(h, pivot)),
                        dome(outer, edge(pivot, t)))
        else:
            return base

    if len(sample) > 2:
        axis = sample[:,0]
        base = np.take(sample, [np.argmin(axis), np.argmax(axis)], axis=0)#get the first and last point [928,0.25][2506,0.36]
        hull = link(dome(sample, base),
                    dome(sample, base[::-1]))
    else:
        hull = sample

    #discard some incorrect points, whose x coordinate is inverse. (2500,xx) (<2500,xx) 
    index_end = 0
    for i in range(len(hull)):
        if hull[i][0] == sample[-1,0]:
            index_end = i
            break
    hull = hull[:index_end+1] 
    hull = pd.DataFrame(hull,columns = ['wavelength', 'reflectance'])
    hull = resample(hull,list(sample[:,0]))
    return hull

# this funtion input an filePath (including file Name and format), and sheet namee, return an DataFrame that save the info of this sheet.
def read_excel(filePath = None, sheetName = 'Sheet1'):
    assert filePath != None or sheetName != None, 'your filePath is not right, program end.\n'
    df_excel = pd.read_excel(filePath, sheetname = sheetName)
    return df_excel

# input an envi file path, return spectrum [[band,ref],......] and its continuum [0.95,0.99,0.26,....]
def get_hull_fromEnvi(filePath = None):
    sp_lib = envi.open(filePath)
    print(sp_lib.names, end = '\n')
    if switch_sp_choice_envi :
        sp_choice = input('input the specturm index you want to fit. (start with 0) \n')
    else:
        sp_choice = 0
    wavelength = sp_lib.bands.centers
    reflectance = sp_lib.spectra[int(sp_choice)]
    spectrum = np.array([wavelength,reflectance]).T
    
    hull = qhull(spectrum)
    return spectrum, hull

# input excel file path, sheet name and pic file index. Return this file's average specturm reading from excel and the continuum of this avg sp. specturm: [[928,0.25], [935,0.41]....]
def get_hull_fromExcel(filePath =None, sheetName ='Sheet1', sp_index = 0):
    df_excel = read_excel(filePath, sheetName = sheetName)
    df_values = df_excel.values 
    list_column = list(df_excel.columns)
    x_begin = list_column.index(928.080017)

    sp = df_values[sp_index][x_begin:]
    wavelength = list_column[x_begin:]
    spectrum = np.array([wavelength,sp]).T
    hull = qhull(spectrum)
    
    return spectrum, hull

#tensorflow function.
def fitting_tf(spectrum, hull, params, fitting_model = MGM):
    # TF graph input
    X = tf.placeholder(tf.float32, shape=[len(spectrum[:,0])])
    Y = tf.placeholder(tf.float32, shape=[len(spectrum[:,1])])

    params_tf = tf.Variable(params)
    # Set parameters
    learning_rate = 0.3
    training_iteration = 3000

    # Construct a  model
    model = fitting_model(spectrum[:,0],list(params))

    # Minimize squared errors, loss function.
    loss_function = tf.reduce_sum((model - Y)**2)

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function) #Gradient descent
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)  

    # Initialize variables
    init = tf.global_variables_initializer()
    #attention, this alg use tensorFlow to fit Gaussian and Hull to sp, what kind of alg did paper use? So
    # Launch a graph
    with tf.Session() as sess:
        sess.run(init)

        display_step = 20
        # Fit all training data
        for iteration in range(training_iteration):
            training_cost, _ = sess.run([loss_function, optimizer], feed_dict={X: list(spectrum[:,0]), Y: list(spectrum[:,1])})

            # Display logs per iteration step
            if iteration % display_step == 0:
                params_new = sess.run(params, feed_dict={X: list(spectrum[:,0]), Y: list(spectrum[:,1])})
                
        # Final parameters
        params_new = sess.run(params, feed_dict={X: list(spectrum[:,0]), Y: list(spectrum[:,1])})


    offset = []
    centers = [params_new[1],params_new[4],params_new[7]]
    for i in centers:
        for j in range(len(hull)-1):
            if (i >= list_hull[j][0] and i < list_hull[j+1][0]):
                offset.append( (i - hull[j][0]) * (hull[j][1] - hull[j+1][1]) / (hull[j][0] - hull[j+1][0]) + hull[j][1] )

    return params_new

#least square fitting function
def fitting_leastSquare(spectrum, params, fitting_model = multi_MGM, hull = 0):

    errFunc = lambda p, x, y: (y - fitting_model(x, p))**2
    
    para_optim, success = optimize.leastsq(errFunc, params, args=(list(spectrum[:,0]), list(spectrum[:,1])), maxfev = 20000)
    return para_optim
#plot figures. Gaussian fitting curve and discret gaussians.
def plot_figures(para_optimize, axis_x, axis_y):
    plt.figure('ori and fitting spectrum')
    Gaussian_num = int(len(para_optimize)/3)
    for i in range(Gaussian_num):
        params_temp = []
        params_temp.append(para_optimize[i])
        params_temp.append(para_optimize[Gaussian_num+i])
        params_temp.append(para_optimize[2*Gaussian_num+i])
        params_temp.append(para_optimize[-1])
        plt.plot(axis_x, gaussian(axis_x, params_temp),label = str(params_temp[2]))
    plt.plot(axis_x, axis_y, lw=2, c='g',label='Bastnas band1 ori')
    plt.plot(axis_x, multi_MGM( axis_x,para_optimize),lw=0.5, c='r', label='Bastnas band1 fit of 6 MGM')

    diff = multi_MGM( axis_x,para_optimize) - axis_y

    plt.plot(axis_x, diff, label = 'Difference')
    RMS = float(np.sqrt(np.mean( np.array(diff)**2)))
    plt.legend()
    plt.show()
    print(para_optimize,end = '\n')
    print('RMS:%f\t\t mean RMS(percent): %f ' % ((RMS),((RMS)/ np.mean(axis_y)*100)) )

def output_params(params_initial,params_optimize,axis_x,axis_y, band_index = 0):
    filePath = 'data/'
    file_out = open(filePath + 'bastnas_gau_params.txt','a')
    file_out.write('band%d: %fnm - %fnm\n' % (band_index, axis_x[0], axis_x[-1]))
    file_out.write('initial params: \n')
    for item in params_initial:
        file_out.write(str(item) + '\t')
    file_out.write('\noptimize params: \n')
    for item in params_optimize:
        file_out.write(str(item) + '\t')

    diff = multi_MGM( axis_x,para_optimize) - axis_y
    RMS = float(np.sqrt(np.mean( np.array(diff)**2)))
    file_out.write('\nRMS: %f  percent: %f \n\n' % (RMS,(RMS)/ np.mean(axis_y)*100) )

    file_out.close()
    return 1
    
if __name__ == '__main__':
    filePath = 'data/'
    
    if switch_excel  ==1:
        fileName = 'Escondida.xlsx'
        #core continuum funtion, return the ori spectrum and the continuum. sp_index in escondida is 0-80(totally 81 pics' avg spectrum in it.)
        spectrum, hull = get_hull_fromExcel(filePath = filePath+ fileName, sheetName = 'Sheet1', sp_index = 2)
    if switch_envi == 1:
        fileName = 'SpectraForAbsorptionFitting.hdr'
        spectrum, hull = get_hull_fromEnvi(filePath = filePath + fileName)
    #cal the ratio (continuum removal.)
    ratio_continuum = spectrum[:,1] / hull

    #Below is an example how to use ModifiedGaussianModel.py to get the optimize parameters.
    ABP_bands = [(705.619995,770.429993), (770.429993,833.48999),(854.119995,880.380005),(880.380005, 895.549988)]
    ABP_index = []
    spectrum_band = []
    hull_band = []
    for band in ABP_bands:
        (band_begin,band_end) = (list(spectrum[:,0]).index(band[0]),list(spectrum[:,0]).index(band[1]) )
        ABP_index.append((band_begin,band_end))
        spectrum_band.append( spectrum[band_begin:band_end])
        hull_band.append(hull[band_begin:band_end])
    
    #attention, tensorflow's traning result is nan...... I give up this. and use other method to do fitting.
    #fitting_tf(spectrum_band[0], hull_band[0], fitting_model = multi_MGM)
    band_index = 3
    input_band = spectrum_band[band_index]
    axis_x = spectrum_band[band_index][:,0]
    axis_y = spectrum_band[band_index][:,1]

    #set initial params.
    if switch_mutiBands:

        center = [889]
        height = [0.2 for i in range(len(center))]
        width = [5. for i in range(len(center))]
        

        yshift = [0]

        params = []
        params.extend(height)
        params.extend(width)
        params.extend(center)
        params.extend(yshift)
        para_optimize = fitting_leastSquare(input_band, params, fitting_model = multi_MGM, hull = hull_band[0])
        #output_params(params, para_optimize, axis_x ,axis_y, band_index = band_index + 1)
    plot_figures(para_optimize,axis_x, axis_y)

    

    #height 0.01, RMS .000002 / 8; 0.02 .000099; 0.1 .000192 .000712; -0.01, .000618 .002277
    #height 5, RMS .000108/ 399; 
