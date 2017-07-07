import pandas as pd 
import numpy
from pandas import ExcelWriter
from pandas import ExcelFile
import pylab as pl
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d
import math


# Output those specture pictures.
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

def hull_to_spectrum(hull, wavelengths):
    f = interp1d(hull['wavelength'], hull['intensity'])
    return f(wavelengths)

if __name__=="__main__":
    col_name = df.columns
    name_list = list(col_name)
    xbegin = name_list.index(928.080017)
    wavelengths = name_list[xbegin:]

    col_value = df.values
    value_list = list(col_value)
    fig, axs = plt.subplots(10,4, figsize=(10,5)) # multiple plots in one figure
    for number in range(40, 80):
       
        spectrum = col_value[number][xbegin:]
        sample = numpy.array([wavelengths,spectrum]).T
        #print(sample)

        hull = qhull(sample)
        hull = pd.DataFrame(hull,columns=['wavelength','intensity'])


        ax = math.floor(number/4)-10
        # axs[ax][0].plot(wavelengths, spectrum, c='b')
        # axs[ax][0].plot(hull.iloc[:-1]['wavelength'], hull.iloc[:-1]['intensity'], color='k')

        hull_spectrum = hull_to_spectrum(hull[:-1], wavelengths)
        spectrum2 = spectrum / hull_spectrum
        axs[ax][number%4].plot(spectrum2)
        # pl.legend()
    # fig.tight_layout()
    plt.show()
