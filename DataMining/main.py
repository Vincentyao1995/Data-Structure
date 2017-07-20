import pandas as pd 
import numpy as np
from pandas import ExcelWriter
from pandas import ExcelFile
import tensorflow as tf
import matplotlib.pylab as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d
from scipy import signal

# Finding the envelop of the spectrum
def qhull(sample):
    link = lambda a,b: np.concatenate((a,b[1:]))
    edge = lambda a,b: np.concatenate(([a],[b]))

    def dome(sample,base): 
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
        base = np.take(sample, [np.argmin(axis), np.argmax(axis)], axis=0)
        return link(dome(sample, base),
                    dome(sample, base[::-1]))
    else:
        return sample

# Read the file and store in pandas
df = pd.read_excel('Escondida_Combined.xlsx', sheetname='EscondidaMultiSensorDataset')

def normalize(array):
    # return (array - array.mean()) / array.std()
    return array

# Whether to set the proportion and how to set 
def sigmod(x):
    # return 1.0 / (1 + tf.exp(-x))
    return x

def plotsigmod(x):
    # return 1.0 / (1 + np.exp(-x))
    return x

def gaussian(x, height, center, width):
    return height*tf.exp(-1.0*(x - center)**2/(2*width**2)) 

# Use lorentzian function 
def lorentzian(x, height, center, width):
    """ defined such that height is the height when x==x0 """
    halfWSquared = (width/2.)**2
    return (height * halfWSquared) / ((x - center)**2 + halfWSquared)

def gauss_loren(x, height, center, width, rate):
    sigmodrate = sigmod(rate)
    return sigmodrate * gaussian(x, height, center, width) + (1-sigmodrate) * lorentzian(x, height, center, width)
    # return rate * gaussian(x, height, center, width) + (1-rate) * lorentzian(x, height, center, width)

def three_combinedfunc(x,h1,c1,w1,h2,c2,w2,h3,c3,w3,r):
    return (gauss_loren(x,h1,c1,w1,r)+gauss_loren(x,h2,c2,w2,r)+gauss_loren(x,h3,c3,w3,r))

# Use the sum of three lorentzian function
def three_lorentzian(x, h1,c1,w1,h2,c2,w2,h3,c3,w3):
    return (lorentzian(x,h1,c1,w1)+lorentzian(x,h2,c2,w2)+lorentzian(x,h3,c3,w3))

def three_gaussian(x, h1,c1,w1,h2,c2,w2,h3,c3,w3):
    return (gaussian(x,h1,c1,w1)+gaussian(x,h2,c2,w2)+gaussian(x,h3,c3,w3))

def plotgaussian(x, height, center, width):
    return height*np.exp(-1.0*(x - center)**2/(2*width**2)) 
    # Use different rate to get the more precise values.
def plotgauss_loren(x, height, center, width, rate):
    sigmodrate = plotsigmod(rate)
    return sigmodrate * plotgaussian(x, height, center, width) + (1-sigmodrate) * lorentzian(x, height, center, width)

def plotthree_combinedfunc(x,h1,c1,w1,h2,c2,w2,h3,c3,w3,r):
    return (plotgauss_loren(x,h1,c1,w1,r)+plotgauss_loren(x,h2,c2,w2,r)+plotgauss_loren(x,h3,c3,w3,r))

# To interp the envelop
def hull_to_spectrum(hull, wavelengths):
    f = interp1d(hull['wavelength'], hull['intensity'])
    return f(wavelengths)

# Training the spectrum to fit the curves with tensorflow
def training(pwavelengths, pspectrum, phull_spectrum, pointsarray):

    # the ranges of the segment with two points and the center, it is near the center parts.
    pcenter, pbegin, pend = pointsarray
    pc1 = ((pwavelengths[pbegin] + 2 * pwavelengths[pcenter]) / 3.).astype(np.float32)
    pc2 = pwavelengths[pcenter].astype(np.float32)
    pc3 = ((2 * pwavelengths[pcenter] + pwavelengths[pend]) / 3.).astype(np.float32)

    print (pc1,pc2,pc3)

    wavelengths = pwavelengths[pbegin:pend+1]
    spectrum = pspectrum[pbegin:pend+1]
    hull_spectrum = phull_spectrum[pbegin:pend+1]

    # # TF graph input
    X = tf.placeholder(tf.float32, shape=[len(wavelengths)])
    Y = tf.placeholder(tf.float32, shape=[len(spectrum)])

    # Create a model

    # Set model weights 
    
    # # height = tf.Variable(numpy.random.randn(), name="height")
    # # center = tf.Variable(numpy.random.randn(), name="center")
    # # width = tf.Variable(numpy.random.randn(), name="width")
    h1=tf.Variable(0.,name="h1")
    c1=tf.Variable(tf.convert_to_tensor(pc1,dtype=tf.float32) ,name="c1")
    w1=tf.Variable(5.0,name="w1")
    h2=tf.Variable(0.,name="h2")
    c2=tf.Variable(tf.convert_to_tensor(pc2,dtype=tf.float32) ,name="c2")
    w2=tf.Variable(5.0,name="w2")
    h3=tf.Variable(0.,name="h3")
    c3=tf.Variable(tf.convert_to_tensor(pc3,dtype=tf.float32) ,name="c3")
    w3=tf.Variable(5.0,name="w3")
    r =tf.Variable(0.,name="rate")
    params = [h1,c1,w1,h2,c2,w2,h3,c3,w3,r]

    # # Set parameters
    learning_rate = 0.1
    training_iteration = 3000

    # # Construct a  model
    model = hull_spectrum + three_combinedfunc(X, h1,c1,w1,h2,c2,w2,h3,c3,w3,r)

    # # Minimize squared errors
    cost_function = tf.reduce_sum(tf.pow(model - Y, 2)) #L2 loss
    # # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function) #Gradient descent
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)  

    # Initialize variables
    init = tf.global_variables_initializer()

    def display(iteration, display_step):
        # Display logs per iteration step
        if iteration % display_step == 0:
            _h1,_c1,_w1,_h2,_c2,_w2,_h3,_c3,_w3,_r = sess.run(params, feed_dict={X: wavelengths, Y: spectrum})

            print('Iteration: %04d, cost=%0.9f. [h1,c1,w1]: %0.3f,%0.3f,%0.3f [h2,c2,w2]: %0.3f,%0.3f,%0.3f [h3,c3,w3]: %0.3f,%0.3f,%0.3f rate: %0.3f sigmodrate: %0.3f'  % (
                iteration + 1, 
                training_cost,
                _h1,_c1,_w1,_h2,_c2,_w2,_h3,_c3,_w3,_r,plotsigmod(_r)
            ))

        # Display fitted spectrum occasionally
        if iteration % 500 == 0:
            plt.figure()
            plt.plot(wavelengths, spectrum, c='k', label='observed spectrum')
            plt.plot(wavelengths, hull_spectrum + plotthree_combinedfunc(wavelengths, _h1,_c1,_w1,_h2,_c2,_w2,_h3,_c3,_w3,_r), c='r', label='Fitted line')
            plt.legend()
            plt.show()

    def printParmas():
        _h1,_c1,_w1,_h2,_c2,_w2,_h3,_c3,_w3,_r = sess.run(params, feed_dict={X: wavelengths, Y: spectrum})
        print('Training completed: cost=%0.9f. [h1,c1,w1]: %0.3f,%0.3f,%0.3f [h2,c2,w2]: %0.3f,%0.3f,%0.3f [h3,c3,w3]: %0.3f,%0.3f,%0.3f rate: %0.3f sigmodrate: %0.3f' % (
            training_cost,
            _h1,_c1,_w1,_h2,_c2,_w2,_h3,_c3,_w3,_r,plotsigmod(_r)
        ))
        return _h1,_c1,_w1,_h2,_c2,_w2,_h3,_c3,_w3,_r

    def finalShow():
        plt.figure()
        _h1,_c1,_w1,_h2,_c2,_w2,_h3,_c3,_w3,_r = sess.run(params, feed_dict={X: wavelengths, Y: spectrum})
        # _h1,_c1,_w1,_h2,_c2,_w2,_h3,_c3,_w3,_r = printParmas()
        plt.plot(wavelengths, spectrum, c='k', label='observed spectrum')
        plt.plot(wavelengths, hull_spectrum + plotthree_combinedfunc(wavelengths, _h1,_c1,_w1,_h2,_c2,_w2,_h3,_c3,_w3,_r), c='r', label='Fitted line')
        # Show the three functions
        plt.plot(wavelengths,hull_spectrum + plotgauss_loren(wavelengths, _h1, _c1, _w1, _r), linewidth=2, label='Function1')
        plt.plot(wavelengths,hull_spectrum + plotgauss_loren(wavelengths, _h2, _c2, _w2, _r), linewidth=2, label='Function2')
        plt.plot(wavelengths,hull_spectrum + plotgauss_loren(wavelengths, _h3, _c3, _w3, _r), linewidth=2, label='Function3')

        # display errors and find the most influence points
        # plt.figure()
        # plt.plot(wavelengths, spectrum - hull_spectrum - plotthree_combinedfunc(wavelengths, _h1,_c1,_w1,_h2,_c2,_w2,_h3,_c3,_w3,_r), label='Error')
        # displaymeanval(wavelengths, spectrum - hull_spectrum -plotthree_combinedfunc(wavelengths, _h1,_c1,_w1,_h2,_c2,_w2,_h3,_c3,_w3,_r))
        plt.legend()

    # Launch a graph
    with tf.Session() as sess:
        sess.run(init)

        display_step = 20
        # Fit all training data
        for iteration in range(training_iteration):
            training_cost, _ = sess.run([cost_function, optimizer], feed_dict={X: wavelengths, Y: spectrum})

            # display(iteration, display_step)
            
            
        # Final parameters
        _h1,_c1,_w1,_h2,_c2,_w2,_h3,_c3,_w3,_r = printParmas()

        # Show the final plot
        finalShow()
    


    return np.array([wavelengths, hull_spectrum + plotthree_combinedfunc(wavelengths, _h1,_c1,_w1,_h2,_c2,_w2,_h3,_c3,_w3,_r)])

# smoothing to find the real peaks of the original spectrum
def findpeak(spectrum):
    newspectrum = signal.savgol_filter(spectrum,11,4)
    meanval = newspectrum.mean()
    plt.figure()
    peakind=signal.argrelmin(newspectrum)
    # print (peakind, wavelengths[peakind], newspectrum[peakind])
    plt.plot(wavelengths,newspectrum,c='b')
    

    peakind2=signal.argrelmax(newspectrum)
    # print(peakind2,wavelengths[peakind2],newspectrum[peakind2])

    plt.plot(wavelengths, np.repeat(meanval,len(wavelengths)))
    plt.plot(wavelengths[peakind2], newspectrum[peakind2])
    plt.scatter(wavelengths[peakind], newspectrum[peakind], c='r')
    plt.scatter(wavelengths[peakind2],newspectrum[peakind2],c='g')
    plt.plot(hull.iloc[:-1]['wavelength'], hull.iloc[:-1]['intensity'], color='k')

# display those peaks points and related points and return the values with the removed spectrum
def displaymeanval(wavelengths, spectrum2):
    spectrum2 = signal.savgol_filter(spectrum2,11,4)
    meanval = spectrum2.mean()
    plt.figure()
    peakind=signal.argrelmin(spectrum2)

    # print (peakind) #tuple class
    # print (peakind, wavelengths[peakind], spectrum2[peakind])
    plt.plot(wavelengths,spectrum2,c='b')

    peakind2=signal.argrelmax(spectrum2)
    # print(peakind2,wavelengths[peakind2],spectrum2[peakind2])

    goodpeaks = peakind[0][spectrum2[peakind]<meanval]
    # peakind is tuple, so it is 2 dimension
    # print (goodpeaks) 

    mix = sorted(list(peakind[0][:])+list(peakind2[0][:])+list([0,255]))
    # print (mix)
    mix = np.array(mix)

    usefulinfo = list()
    for index in goodpeaks:
        # print (mix[mix>index])
        # print (mix[mix<index])
        info = np.array([index, mix[mix<index][-1], mix[mix>index][0]])
        usefulinfo.append(info)
    # print (usefulinfo)

    plt.plot(wavelengths, np.repeat(meanval,len(wavelengths)))
    plt.scatter(wavelengths[peakind], spectrum2[peakind], c='r')
    plt.scatter(wavelengths[peakind2],spectrum2[peakind2],c='g')

    return usefulinfo

# change the hull to fit all the data within [0-1]
def changehull(hull):
    c = 0
    for index in range(len(hull)):
        if hull[index][0] == name_list[-1]:
            c = index
            break
    hull = hull[:c+1]
    hull = np.vstack((hull, hull[0]))
    
    return hull
   
# output all the predicted values
def outputval(hull):
    
    list_hull = hull

    offset=list()
    centers = [_c1,_c2,_c3]
    for i in centers:
        for j in range(len(hull)-1):
            if (i>=list_hull[j][0] and i<list_hull[j+1][0]):
                offset.append( (i - list_hull[j][0]) * (list_hull[j][1] - list_hull[j+1][1]) / (list_hull[j][0] - list_hull[j+1][0]) + list_hull[j][1] )
    print (offset+three_lorentzian(centers, _h1,_c1,_w1,_h2,_c2,_w2,_h3,_c3,_w3))

    # print (hull_spectrum)

# display the original curves
def showOriginal(wavelengths, spectrum, spectrum2):
    fig, axs = plt.subplots(1,2, figsize=(10,5)) # multiple plots in one figure

    axs[0].plot(wavelengths, spectrum, c='b')
    axs[0].plot(hull.iloc[:-1]['wavelength'], hull.iloc[:-1]['intensity'], color='k')

    axs[1].plot(wavelengths, spectrum2, c='b')

# display the divide region curves
def showDivideRegion():
    pointsarray = displaymeanval(wavelengths,spectrum2)

    plt.figure()
    plt.plot(wavelengths, spectrum, c='b')
    for index in np.arange(len(pointsarray)):
        pcenter, pbegin, pend = pointsarray[index]
        # print (pcenter, pbegin, pend)
        # plt.plot(wavelengths[pbegin : pend+1],spectrum2[pbegin : pend+1], wavelengths[pbegin : pend+1], spectrum[pbegin : pend+1])
        plt.plot( wavelengths[pbegin : pend+1], spectrum[pbegin : pend+1], linewidth=3)

    plt.show()

# display the fitting region curves
def showFittingRegion():
    results = list()
    pointsarrays = displaymeanval(wavelengths, spectrum2)
    # for index in np.arange(len(pointsarrays)):
    for index in np.arange(0,1):
        pointsarray = pointsarrays[index]
        # print (pc1,pc2,pc3)
        results.append(training(wavelengths, spectrum, hull_spectrum, pointsarray))

    plt.figure()
    plt.plot(wavelengths, spectrum, c='b')
    for index in np.arange(len(results)):
        x,y = results[index]
        plt.plot(x,y,linewidth=3)
    plt.show()

if __name__=="__main__":
    col_name = df.columns
    name_list = list(col_name)
    xbegin = name_list.index(928.080017)
    wavelengths = np.array(name_list[xbegin:])

    col_value = df.values
    value_list = list(col_value)
    spectrum = col_value[41][xbegin:]
   
    sample = np.array([wavelengths,spectrum]).T    
    hull = qhull(sample)

    
    hull = changehull(hull)
    # print (hull)
    hull = pd.DataFrame(hull, columns=['wavelength', 'intensity'])

    hull_spectrum = hull_to_spectrum(hull[:-1], wavelengths)
    spectrum2 = spectrum / hull_spectrum
    
    # showoriginal(wavelengths, spectrum, spectrum2)
    # findpeak(spectrum)
    
    # showDivideRegion()
    showFittingRegion()
    

# [1367.21, 1408.2, 1530.95]
# Results1: 0.000127 0.047,1367.273,1.879; -0.04,1409.746,39.171;  -0.011,1479.427,60.820    

# [1876.67, 1911.17, 2020.8]
# Results2: 0.000586 -0.059,1904.978,24.601; -0.034,1930.458,37.595; -0.037,1976.435,77.011

# [2167.97, 2205.5, 2252.38]
# Results3: 0.000129 -0.014,2163.884,20.079; -0.052,2203.707,35.701; -0.011,2249.93,35.282

    
    plt.show()


# with 3000 iteration: the 42th errors
# 0.0013; 0.00027; 0.00068; 0.0000025; 0; 0.000017;0.00014


# Record:
# 当为r直接相加，并且训练3000次时：其实两千次就可以了，最后rate是0.992
# 当r为sigmod之后进行转换到0-1之间时：其实1500次就可以了，最后sigmodrate是0.998
# 说明还是高斯函数占据主导？