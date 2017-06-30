import pandas as pd 
import numpy as np
from pandas import ExcelWriter
from pandas import ExcelFile
import tensorflow as tf
import matplotlib.pylab as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d

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

df = pd.read_excel('Escondida_Combined.xlsx', sheetname='EscondidaMultiSensorDataset')

def normalize(array):
    # return (array - array.mean()) / array.std()
    return array

def lorentzian(x, height, center, width):
    """ defined such that height is the height when x==x0 """
    halfWSquared = (width/2.)**2
    return (height * halfWSquared) / ((x - center)**2 + halfWSquared)

def three_lorentzian(x, h1,c1,w1,h2,c2,w2,h3,c3,w3):
    return (lorentzian(x,h1,c1,w1)+lorentzian(x,h2,c2,w2)+lorentzian(x,h3,c3,w3))

def hull_to_spectrum(hull, wavelengths):
    f = interp1d(hull['wavelength'], hull['intensity'])
    return f(wavelengths)

if __name__=="__main__":
    col_name = df.columns
    name_list = list(col_name)
    xbegin = name_list.index(928.080017)
    wavelengths = np.array(name_list[xbegin:])

    col_value = df.values
    value_list = list(col_value)
    spectrum = col_value[0][xbegin:]
    sample = np.array([wavelengths,spectrum]).T

    hull = qhull(sample)
    hull = pd.DataFrame(hull, columns=['wavelength', 'intensity'])

    fig, axs = plt.subplots(1,2, figsize=(10,5)) # multiple plots in one figure

    axs[0].plot(wavelengths, spectrum, c='b')
    axs[0].plot(hull.iloc[:-1]['wavelength'], hull.iloc[:-1]['intensity'], color='k')

    hull_spectrum = hull_to_spectrum(hull[:-1], wavelengths)
    spectrum2 = spectrum / hull_spectrum
    axs[1].plot(spectrum2)

    plt.show()

    # TF graph input
    X = tf.placeholder(tf.float32, shape=[len(wavelengths)])
    Y = tf.placeholder(tf.float32, shape=[len(spectrum)])

    # Create a model

    # Set model weights 
    
    # height = tf.Variable(numpy.random.randn(), name="height")
    # center = tf.Variable(numpy.random.randn(), name="center")
    # width = tf.Variable(numpy.random.randn(), name="width")
    h1=tf.Variable(0.,name="h1")
    c1=tf.Variable(1420.0,name="c1")
    w1=tf.Variable(10.0,name="w1")
    h2=tf.Variable(0.,name="h2")
    c2=tf.Variable(1920.0,name="c2")
    w2=tf.Variable(10.0,name="w2")
    h3=tf.Variable(0.,name="h3")
    c3=tf.Variable(2210.0,name="c3")
    w3=tf.Variable(10.0,name="w3")
    params = [h1,c1,w1,h2,c2,w2,h3,c3,w3]

    # Set parameters
    learning_rate = 0.3
    training_iteration = 3000

    # Construct a  model
    model = hull_spectrum + three_lorentzian(X, h1,c1,w1,h2,c2,w2,h3,c3,w3)

    # Minimize squared errors
    cost_function = tf.reduce_sum(tf.pow(model - Y, 2)) #L2 loss
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function) #Gradient descent
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)  

    # Initialize variables
    init = tf.global_variables_initializer()

    # Launch a graph
    with tf.Session() as sess:
        sess.run(init)

        display_step = 20
        # Fit all training data
        for iteration in range(training_iteration):
            training_cost, _ = sess.run([cost_function, optimizer], feed_dict={X: wavelengths, Y: spectrum})

            # Display logs per iteration step
            if iteration % display_step == 0:
                _h1,_c1,_w1,_h2,_c2,_w2,_h3,_c3,_w3 = sess.run(params, feed_dict={X: wavelengths, Y: spectrum})

                print('Iteration: %04d, cost=%0.9f. [h1,c1,w1]: %0.3f,%0.3f,%0.3f [h2,c2,w2]: %0.3f,%0.3f,%0.3f [h3,c3,w3]: %0.3f,%0.3f,%0.3f' % (
                    iteration + 1, 
                    training_cost,
                    _h1,_c1,_w1,_h2,_c2,_w2,_h3,_c3,_w3
                ))

            # Display fitted spectrum occasionally
            if iteration % 500 == 0:
                plt.plot(wavelengths, spectrum, c='k', label='observed spectrum')
                plt.plot(wavelengths, hull_spectrum + three_lorentzian(wavelengths, _h1,_c1,_w1,_h2,_c2,_w2,_h3,_c3,_w3), c='r', label='Fitted line')
                plt.legend()
                plt.show()
            
        # Final parameters
        _h1,_c1,_w1,_h2,_c2,_w2,_h3,_c3,_w3 = sess.run(params, feed_dict={X: wavelengths, Y: spectrum})
        print('Training completed: cost=%0.9f. [h1,c1,w1]: %0.3f,%0.3f,%0.3f [h2,c2,w2]: %0.3f,%0.3f,%0.3f [h3,c3,w3]: %0.3f,%0.3f,%0.3f' % (
            training_cost,
            _h1,_c1,_w1,_h2,_c2,_w2,_h3,_c3,_w3
        ))
    
    # Final plot
    plt.plot(wavelengths, spectrum, c='k', label='observed spectrum')
    plt.plot(wavelengths, hull_spectrum + three_lorentzian(wavelengths, _h1,_c1,_w1,_h2,_c2,_w2,_h3,_c3,_w3), c='r', label='Fitted line')
    plt.legend()
    plt.show()
