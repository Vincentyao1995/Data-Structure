import numpy as np

def lorentzian(x, height=1., center=0., width=1.):
	""" defined such that height is the height when x==x0 """
	halfWSquared = (width/2.)**2
	return (height * halfWSquared) / ((x - center)**2 + halfWSquared)

def gaussian(x, height, center_x, width, yshift):
	return yshift + height * np.exp( - (x - center_x)**2 / (width*2) )