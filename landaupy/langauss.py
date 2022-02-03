import numpy as np
from .landau import pdf as landau_pdf

def gaussian_pdf(x, mu, gauss_sigma):
	return np.exp(-1/2*((x-mu)/gauss_sigma)**2)/gauss_sigma/(2*np.pi)**.5

def pdf_not_vectorized(x: float, mu: float, eta: float, gauss_sigma: float) -> float:
	"""Non vectorized and rustic langaus PDF calculation. **This function should be avoided**, this is almost a copy-paste from the original Root code in https://root.cern.ch/doc/master/langaus_8C.html only for testing purposes."""
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
	"""Langauss probability density function (PDF), i.e. a Landau convoluted 
	with a Gaussian, commonly found when dealing with the interaction of
	charged particles with matter. This function was addapted from 
	[the implementation in Root](https://root.cern.ch/doc/master/langaus_8C.html).

	Parameters
	----------
	x: float, numpy array
		Point (or points) where to calculate the PDF.
	landau_x_mpv: float
		Position of the most probable value (MPV) of the Landau component.
	landau_xi: float
		Parameter $xi$ of the Landau component.
	gauss_sigma: float
		Standard deviation of the Gaussian component.

	Returns
	-------
	langauss_pdf: float, numpy array
		Value of the langauss PDF.
	"""
	if any([not isinstance(arg, (int,float)) for arg in [landau_x_mpv,landau_xi,gauss_sigma]]):
		raise TypeError(f'`landau_x_mpv`, `landau_xi` and `gauss_sigma` must be scalar numbers, they are {type(landau_x_mpv),type(landau_xi),type(gauss_sigma)} respectively.')
	if not isinstance(x, (int, float, np.ndarray)):
		raise TypeError(f'`x` must be either a number or a numpy array, received object of type {type(x)}.')
	if isinstance(x, (int, float)):
		x = np.array([x])
	if gauss_sigma == 0: # There is no Gaussian...
		return landau_pdf(x, landau_x_mpv, landau_xi)
	gaussian_extension = 8 # Number of sigmas to extend around `x` when performing the convolution.
	xlow = x - gaussian_extension*gauss_sigma
	xupp = x + gaussian_extension*gauss_sigma
	xx = np.linspace(xlow, xupp, 111)
	result = np.diff(xx,axis=0)[0]*(landau_pdf(xx.reshape(xx.shape[0]*xx.shape[1]), landau_x_mpv, landau_xi).reshape(xx.shape)*gaussian_pdf(x, xx, gauss_sigma)).sum(axis=0)
	return np.squeeze(result)

def automatic_cdf(x: float, landau_x_mpv: float, landau_xi: float, gauss_sigma: float):
	"""Langauss cumulative distribution function (the integral of the PDF between -inf and x) calculated by brute-force. **This function should be avoided** as it is very slow, only here for testing purposes."""
	from scipy.integrate import quad
	integrand = lambda X: pdf(X, landau_x_mpv, landau_xi, gauss_sigma)
	def _cdf(x):
		return quad(integrand, -float('inf'), x)
	_cdf = np.vectorize(_cdf)
	integral, error = _cdf(x)
	return integral

def cdf(x, landau_x_mpv: float, landau_xi: float, gauss_sigma: float, lower_n_xi_sigma: float=4, dx_n_xi: float=4):
	"""Langauss cumulative distribution function (CDF). 

	Parameters
	----------
	x: float, numpy array
		Point (or points) where to calculate the CDF.
	landau_x_mpv: float
		Position of the most probable value (MPV) of the Landau component.
	landau_xi: float
		Parameter $xi$ of the Landau component.
	gauss_sigma: float
		Standard deviation of the Gaussian component.
	lower_n_xi_sigma: float, default 4
		The numeric integration lower limit is dependent on `xi` in the 
		following way 
		`np.minimum(x, landau_x_mpv - lower_n_xi_sigma*(landau_xi+gauss_sigma))`
		The default value should work in any case but if you find troubles 
		you can change it. Increasing `lower_n_xi` will extend the integration 
		towards -infinity yielding more accurate results but extending 
		the computation time.
	dx_n_xi: float, default 4
		The integration dx is calculated as `dx = xi/dx_n_xi`. The default 
		value should work in any case but you can change it if you see
		issues. Increasing `dx_n_xi` will produce a smaller dx and thus
		more accurate results, but the computation time will also be
		increased.

	Returns
	-------
	landau_cdf: float, numpy array
		Value of the Landau CDF.
	"""
	if any([not isinstance(arg, (int,float)) for arg in [landau_x_mpv,landau_xi,gauss_sigma]]):
		raise TypeError(f'`landau_x_mpv`, `landau_xi` and `gauss_sigma` must be scalar numbers, they are {type(landau_x_mpv),type(landau_xi),type(gauss_sigma)} respectively.')
	if not isinstance(x, (int, float, np.ndarray)):
		raise TypeError(f'`x` must be either a number or a numpy array, received object of type {type(x)}.')
	if isinstance(x, (int, float)):
		x = np.array([x])
	x_low = np.minimum(x, landau_x_mpv - lower_n_xi_sigma*(landau_xi+gauss_sigma)) # At this point the PDF is 1e-9 smaller than in the peak, and goes very quickly to 0.
	x_high = x
	dx = landau_xi/dx_n_xi
	xx = np.linspace(x_low, x_high, int(max(x_high-x_low)/dx))
	xx[xx>x_high] = float('NaN')
	return np.squeeze(
		np.trapz(
			x = xx,
			y = pdf(xx.reshape(xx.shape[0]*xx.shape[1]),landau_x_mpv, landau_xi, gauss_sigma).reshape(xx.shape),
			axis = 0,
		)
	)
