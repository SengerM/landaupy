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

def pdf(x, mu: float, eta: float, sigma: float):
	"""Vectorized version of the langauss PDF function, adapted from https://root.cern.ch/doc/master/langaus_8C.html"""
	if any([not isinstance(arg, (int,float)) for arg in [mu,eta,sigma]]):
		raise TypeError(f'`mu`, `eta` and `sigma` must be scalar numbers, they are {type(mu),type(eta),type(sigma)} respectively.')
	if not isinstance(x, (int, float, np.ndarray)):
		raise TypeError(f'`x` must be either a number or a numpy array, received object of type {type(x)}.')
	if isinstance(x, (int, float)):
		x = np.array([x])
	mpc = mu+0.22278298*eta
	sc = 8
	xlow = x - sc*sigma
	xupp = x + sc*sigma
	xx = np.linspace(xlow, xupp, 111)
	result = np.diff(xx,axis=0)[0]*(landau_pdf(xx.reshape(xx.shape[0]*xx.shape[1]), mpc, eta).reshape(xx.shape)/eta*gaussian_pdf(x, xx, sigma)).sum(axis=0)
	return np.squeeze(result)
