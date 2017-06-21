import pandas as pd 
import numpy
from pandas import ExcelWriter
from pandas import ExcelFile
import pylab as pl
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def qhull(sample):
    link = lambda a,b: numpy.concatenate((a,b[1:]))
    edge = lambda a,b: numpy.concatenate(([a],[b]))

    def dome(sample,base): 
        h, t = base
        dists = numpy.dot(sample-h, numpy.dot(((0,-1),(1,0)),(t-h)))
        outer = numpy.repeat(sample, dists>0, axis=0)
        
        if len(outer):
            pivot = sample[numpy.argmax(dists)]
            return link(dome(outer, edge(h, pivot)),
                        dome(outer, edge(pivot, t)))
        else:
            return base

    if len(sample) > 2:
        axis = sample[:,0]
        base = numpy.take(sample, [numpy.argmin(axis), numpy.argmax(axis)], axis=0)
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


if __name__=="__main__":
    col_name = df.columns
    name_list = list(col_name)
    xbegin = name_list.index(928.080017)
    x = name_list[xbegin:]

    col_value = df.values
    value_list = list(col_value)
    y = col_value[0][xbegin:]
    sample = numpy.array([x,y]).T
    #print(sample)

    hull = qhull(sample)


    for s in sample:
        pl.plot([s[0]], [s[1]], 'b.')

    i = 0
    while i < len(hull)-1:
        pl.plot([hull[i][0], hull[i+1][0]], [hull[i][1], hull[i+1][1]], color='k' )
        i = i + 1

    #print (len(hull))
    #print (hull)
    off=list()

    for i in x:
        for j in range(len(hull)-1):
            if (i >= hull[j][0] and i <hull[j+1][0]):
                off.append( (i - hull[j][0]) * (hull[j][1] - hull[j+1][1]) / (hull[j][0] - hull[j+1][0]) + hull[j][1] )
    off.append(hull[-2][1])
    
    
    plt.figure(1)
    pl.plot([hull[-1][0], hull[0][0]], [hull[-1][1], hull[0][1]], color='k')
    plt.figure()
    y_=off/y-1
    pl.plot(x,y_)

    # f = open('data.txt', 'w')
    # f.write(str(off))
    # f.close()



    x_sample = numpy.array(x)
    y_sample = numpy.array(y_)

    # numpy.savetxt("y_.txt", y_sample)

    x_data = normalize(x_sample)
    y_data = normalize(y_sample)
    samples_number = x_data.size

    # TF graph input
    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    # Create a model

    # Set model weights 
    
    # height = tf.Variable(numpy.random.randn(), name="height")
    # center = tf.Variable(numpy.random.randn(), name="center")
    # width = tf.Variable(numpy.random.randn(), name="width")
    h1=tf.Variable(0.13,name="h1")
    c1=tf.Variable(1420.0,name="c1")
    w1=tf.Variable(5.0,name="w1")
    h2=tf.Variable(0.25,name="h2")
    c2=tf.Variable(1920.0,name="c2")
    w2=tf.Variable(5.0,name="w2")
    h3=tf.Variable(0.20,name="h3")
    c3=tf.Variable(2210.0,name="c3")
    w3=tf.Variable(3.0,name="w3")

    # Set parameters
    learning_rate = 1
    training_iteration = 2000

    # Construct a  model
    
    model = three_lorentzian(X, h1,c1,w1,h2,c2,w2,h3,c3,w3)

    # Minimize squared errors
    cost_function = tf.reduce_sum(tf.pow(model - Y, 2))/(2 * samples_number) #L2 loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function) #Gradient descent

    # Initialize variables
    init = tf.initialize_all_variables()

    # Launch a graph
    with tf.Session() as sess:
        sess.run(init)

        display_step = 20
     # Fit all training data
        for iteration in range(training_iteration):
            for (x, y) in zip(x_data, y_data):
                sess.run(optimizer, feed_dict={X: x, Y: y})

        #     xh1,xc1,xw1,xh2,xc2,xw2,xh3,xc3,xw3 = sess.run([h1,c1,w1,h2,c2,w2,h3,c3,w3])

        #     plt.plot(x_data, three_lorentzian(x_data, xh1,xc1,xw1,xh2,xc2,xw2,xh3,xc3,xw3), label='Fitted line')
        
        #     plt.legend()

        # plt.show()

            # Display logs per iteration step
            if iteration % display_step == 0:
                print ("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(sess.run(cost_function, feed_dict={X:x_data, Y:y_data})),\
                    "[h1,c1,w1]: ",sess.run([h1,c1,w1]), " [h2,c2,w2]: ",sess.run([h2,c2,w2]), " [h3,c3,w3]: ",sess.run([h3,c3,w3])
                )
            
        tuning_cost = sess.run(cost_function, feed_dict={X: normalize(x_data), Y: normalize(y_data)})
            
        print ("Tuning completed:", "cost=", "{:.9f}".format(tuning_cost), "[h1,c1,w1]: ",sess.run([h1,c1,w1]), " [h2,c2,w2]: ",sess.run([h2,c2,w2]), " [h3,c3,w3]: ",sess.run([h3,c3,w3]))
    
        # Validate a tuning model
    
        testing_cost = sess.run(cost_function, feed_dict={X: x_data, Y: y_data})
    
        print ("Testing data cost:" , testing_cost)
    
        # Display a plot
        # plt.figure()
        # plt.plot(x_data, y_data, 'ro', label='Normalized samples')
        #plt.plot(size_data_test_n, price_data_test_n, 'go', label='Normalized testing samples')

        xh1,xc1,xw1,xh2,xc2,xw2,xh3,xc3,xw3 = sess.run([h1,c1,w1,h2,c2,w2,h3,c3,w3])

        plt.plot(x_data, three_lorentzian(x_data, xh1,xc1,xw1,xh2,xc2,xw2,xh3,xc3,xw3), label='Fitted line')
        
        plt.legend()

    pl.show()



