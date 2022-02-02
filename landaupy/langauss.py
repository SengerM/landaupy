import numpy as np
from .landau import pdf as landau_pdf

def gaussian_pdf(x, mu, sigma):
	return np.exp(-1/2*((x-mu)/sigma)**2)/sigma/(2*np.pi)**.5

def pdf_not_vectorized(x: float, mu: float, eta: float, sigma: float) -> float:
	"""Non vectorized and rustic langaus PDF calculation. This function should be avoided, this is almost a copy-paste from the original Root code in https://root.cern.ch/doc/master/langaus_8C.html only for testing purposes."""
	if any([not isinstance(arg, (int,float)) for arg in [x,mu,eta,sigma]]):
		raise TypeError('All arguments must be float numbers.')
	
	mpshift = -0.22278298 # Landau maximum location shift in original code is wrong, since the shift does not depend on mu only

	np = 100 # number of convolution steps
	sc = 8 # convolution extends to +-sc Gaussian sigmas

	# Convolution steps have to be increased if sigma > eta * 5 to get stable solution that does not oscillate, addresses #1
	if sigma > 3 * eta:
		np *= int(sigma / eta / 3)
	if np > 100000: # Do not use too many convolution steps to save time
		np = 100000

	# MP shift correction
	mpc = mu - mpshift * eta

	# Range of convolution integral
	xlow = x - sc * sigma
	xupp = x + sc * sigma

	step = (xupp - xlow) / np

	# Discrete linear convolution of Landau and Gaussian
	suma = 0
	for i in range(1, int(np+1)):
		xx = xlow + (i - 0.5) * step
		fland = landau_pdf(xx, mpc, eta) / eta
		suma += fland * gaussian_pdf(x, xx, sigma)

	return step*suma

def pdf_convolution_vectorized(x: float, mu: float, eta: float, sigma: float):
	if any([not isinstance(arg, (int,float)) for arg in [x,mu,eta,sigma]]):
		raise TypeError('All arguments must be float numbers.')
	convolution_steps = 100 if sigma<3*eta else int(100*sigma/eta/3) # In the original code this is called `np` but this name is usually reserved for numpy in Python.
	if convolution_steps > 99999: 
		convolution_steps = 99999
	convolution_steps = int(convolution_steps)
	mpc = mu+0.22278298*eta
	sc = 8
	xlow = x - sc*sigma
	xupp = x + sc*sigma
	step = (xupp-xlow)/convolution_steps
	xx = np.arange(xlow,xupp,step)
	result = step*sum(landau_pdf(xx, mpc, eta)/eta*gaussian_pdf(x, xx, sigma))
	return result
