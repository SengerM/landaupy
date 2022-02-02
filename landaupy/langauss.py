import numpy as np
from .landau import pdf as landau_pdf

def gaussian_pdf(x, mu, gauss_sigma):
	return np.exp(-1/2*((x-mu)/gauss_sigma)**2)/gauss_sigma/(2*np.pi)**.5

def pdf_not_vectorized(x: float, mu: float, eta: float, gauss_sigma: float) -> float:
	"""Non vectorized and rustic langaus PDF calculation. This function should be avoided, this is almost a copy-paste from the original Root code in https://root.cern.ch/doc/master/langaus_8C.html only for testing purposes."""
	if any([not isinstance(arg, (int,float)) for arg in [x,mu,eta,gauss_sigma]]):
		raise TypeError('All arguments must be float numbers.')
	
	mpshift = -0.22278298 # Landau maximum location shift in original code is wrong, since the shift does not depend on mu only

	np = 100 # number of convolution steps
	sc = 8 # convolution extends to +-sc Gaussian sigmas

	# Convolution steps have to be increased if gauss_sigma > eta * 5 to get stable solution that does not oscillate, addresses #1
	if gauss_sigma > 3 * eta:
		np *= int(gauss_sigma / eta / 3)
	if np > 100000: # Do not use too many convolution steps to save time
		np = 100000

	# MP shift correction
	mpc = mu - mpshift * eta

	# Range of convolution integral
	xlow = x - sc * gauss_sigma
	xupp = x + sc * gauss_sigma

	step = (xupp - xlow) / np

	# Discrete linear convolution of Landau and Gaussian
	suma = 0
	for i in range(1, int(np+1)):
		xx = xlow + (i - 0.5) * step
		fland = landau_pdf(xx, mpc, eta) / eta
		suma += fland * gaussian_pdf(x, xx, gauss_sigma)

	return step*suma

def pdf(x, landau_x_mpv: float, landau_xi: float, gauss_sigma: float):
	"""Vectorized version of the langauss PDF function, adapted from https://root.cern.ch/doc/master/langaus_8C.html
	x: float, numpy array. Value where to calculate the PDF.
	landau_x_mpv: float. Location of the peak of the Landau component, i.e. the most probable value (MPV) of the Landau. Note that after the convolution with the Gaussian the MPV of the langauss will not be at landau_x_mpv but slightly shifted towards a higher value.
	landau_xi: float. Width of the Landau component.
	gauss_sigma: float. Sigma of the Gaussian component."""
	if any([not isinstance(arg, (int,float)) for arg in [landau_x_mpv,landau_xi,gauss_sigma]]):
		raise TypeError(f'`landau_x_mpv`, `landau_xi` and `gauss_sigma` must be scalar numbers, they are {type(landau_x_mpv),type(landau_xi),type(gauss_sigma)} respectively.')
	if not isinstance(x, (int, float, np.ndarray)):
		raise TypeError(f'`x` must be either a number or a numpy array, received object of type {type(x)}.')
	if isinstance(x, (int, float)):
		x = np.array([x])
	gaussian_extension = 8 # Number of sigmas to extend around `x` when performing the convolution.
	xlow = x - gaussian_extension*gauss_sigma
	xupp = x + gaussian_extension*gauss_sigma
	xx = np.linspace(xlow, xupp, 111)
	result = np.diff(xx,axis=0)[0]*(landau_pdf(xx.reshape(xx.shape[0]*xx.shape[1]), landau_x_mpv, landau_xi).reshape(xx.shape)*gaussian_pdf(x, xx, gauss_sigma)).sum(axis=0)
	return np.squeeze(result)
