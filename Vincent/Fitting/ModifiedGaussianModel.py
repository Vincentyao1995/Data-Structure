import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import spectral.io.envi as envi
import tensorflow as tf


switch_excel = 0
switch_envi = 1
switch_dataFrame = 0

#Modified gaussian Model
def MGM(x,height, center, width, yshift, n = 1):
	return yshift + height * np.exp( - (x**n - center**n)**2 / (width*2) )

#Multiple MGM, input a para list, then they would use the list to construct muti- Gaussian
def muti_MGM(x,height,center,width,yshift, n = 1):
    assert len(height) == len(center) == len(width) and len(yshift) == 1, 'your Muti-MGM paras is not equa, please check out.'
    
    for i in range(len(height)):
        newGassuian = MGM(x,height[i],center[i],width[i],yshift = 0)
        res += newGassuian
        if i == len(height) - 1:
            res += yshift 
    return res

#original gaussian function
def gaussian(x, height, center_x, width, yshift):
	return yshift + height * np.exp( - (x - center_x)**2 / (width*2) )

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

    sp_choice = input('input the specturm index you want to fit. (start with 0) \n')

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
def fitting_tf(spectrum, hull, fitting_model = MGM):
    # TF graph input
    X = tf.placeholder(tf.float32, shape=[len(spectrum[:,0])])
    Y = tf.placeholder(tf.float32, shape=[len(spectrum[:,1])])

    
    #height = tf.Variable(0.1, name = 'height')
    #width = tf.Variable(10.0, name  = 'width')
    #center = tf.Variable(550.0, name = 'center')

    height = [0.65,0.7,0.75,0.57,0.48,0.34]
    width = [0.01 for i in range(6)]
    center = [732,736,740,750,753,759]
    for i in range(len(height)):
        params.append(height[i])
        params.append(center[i])
        params.append(width[i])

    # Set parameters
    learning_rate = 0.3
    training_iteration = 3000

    # Construct a  model
    model = hull_spectrum + fitting_model(X,height,center,width, 0, n = -1)# attention, use the code until here. this is where to define model. hull_sp + Gassuian(init_paras). use qhull function to get the hull of a spectrum.

    # Minimize squared errors, loss function.
    loss_function = tf.reduce_sum(tf.pow(model - Y, 2))

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

    print (offset + fitting_model(centers,height,center,width,0))
    outputfile.append(dict({number:( offset + fitting_model(centers, height,center, width))})) 

#plot figures.
def plot_figures():
    plt.figure('SP_lib_index03')
    plt.plot(spectrum[:,0], spectrum[:,1], '-')
    plt.plot(spectrum[:,0], ratio_continuum)
    plt.plot(spectrum[:,0], hull)
    plt.show()

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
    #plot_figures()
    #tensorflow to fitting. input Model u want to use and spectrum band u want to fit.   
    ABP_bands = [(705.619995,770.429993), (770.429993,833.48999),(833.48999,880.380005),(880.380005,900.349976)]
    ABP_index = []
    spectrum_band = []
    hull_band = []
    for band in ABP_bands:
        (band_begin,band_end) = (spectrum[:,0].index(band[0]),spectrum[:,0].index(band[1]) )
        ABP_index.append((band_begin,band_end))
        spectrum_band.append( spectrum[band_begin:band_end])
        hull_band.append(hull[band_begin:band_end])

    fitting_tf(spectrum_band[0], hull_band[0], fitting_model = muti_MGM)
    
