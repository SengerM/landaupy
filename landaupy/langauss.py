import numpy as np
from . import landau
from .samplers import sample_distribution_given_cdf
from . import _check_types as ct
from scipy.special import erf

def gaussian_pdf(x, mu, sigma):
	if sigma<=0:
		return x*float('NaN')
	else:
		return np.exp(-1/2*((x-mu)/sigma)**2)/sigma/(2*np.pi)**.5

def pdf_not_vectorized(x: float, mu: float, eta: float, gauss_sigma: float) -> float:
	"""Non vectorized and rustic langaus PDF calculation. **This function 
	should be avoided**, this is almost a copy-paste from [the original 
	Root code](https://root.cern.ch/doc/master/langaus_8C.html) only for
	testing purposes.
	"""
	ct.check_are_instances({'x':x,'mu':mu,'eta':eta,'gauss_sigma':gauss_sigma}, (int, float))
	
	mpshift = 0 #-0.22278298 # Landau maximum location shift in original code is wrong, since the shift does not depend on mu only

	np = 100 # number of convolution steps
	sc = 5 # convolution extends to +-sc Gaussian sigmas

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
		fland = landau.pdf_not_vectorized(xx, mpc, eta) # Here there was a division by `eta` that I removed because it produces a function that normalizes to 1/eta instead of 1.
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
	
	Error handling
	--------------
	Rises `TypeError` if the parameters are not within the accepted types.
	Non valid values (e.g. sigma<0) rise no errors but return `float('NaN')`.
	
	Notes
	-----
	For certain values of the parameters an approximation is returned. This
	is to avoid numerical instabilities and to increase the calculation
	speed. Such approximations are:
	- If `gauss_sigma < 1e-6*landau_xi` it returns a Landau.
	- If `gauss_sigma > 10000*landau_xi` it returns a Gaussian.
	"""
	ct.check_are_instances({'landau_x_mpv':landau_x_mpv, 'landau_xi':landau_xi, 'gauss_sigma':gauss_sigma}, (int, float))
	ct.check_is_instance(x, 'x', (int, float, np.ndarray))
	if isinstance(x, (int, float)):
		x = np.array([x])
	if gauss_sigma < 0 or landau_xi < 0 or gauss_sigma == landau_xi == 0: # Non valid values.
		result = x*float('NaN')
	else: # sigma>=0 and xi>=0 but sigma != xi
		if gauss_sigma < 1e-6*landau_xi: # The Gaussian contribution is negligible...
			result = landau.pdf(x, landau_x_mpv, landau_xi)
		elif gauss_sigma > 10000*landau_xi: # The Landau contribution is negligible...
			result = gaussian_pdf(x, landau_x_mpv, gauss_sigma)
		else: # All cases where Landau and Gauss contributions are important simultaneously...
			gaussian_extension = 5 # Number of sigmas to extend around `x` when performing the convolution.
			xlow = x - gaussian_extension*gauss_sigma
			xupp = x + gaussian_extension*gauss_sigma
			n_values_convolution_axis = 100
			if gauss_sigma > landau_xi*3:
				n_values_convolution_axis *= int(gauss_sigma/landau_xi/3) # To avoid instabilities, see https://github.com/SiLab-Bonn/pylandau/issues/1
			n_values_convolution_axis = min(n_values_convolution_axis, 100000) # Limit the maximum value.
			xx = np.linspace(xlow, xupp, n_values_convolution_axis)
			result = np.diff(xx,axis=0)[0]*(landau.pdf(xx.reshape(xx.shape[0]*xx.shape[1]), landau_x_mpv, landau_xi).reshape(xx.shape)*gaussian_pdf(x, xx, gauss_sigma)).sum(axis=0)
	return np.squeeze(result)

def automatic_cdf(x, landau_x_mpv: float, landau_xi: float, gauss_sigma: float):
	"""Langauss cumulative distribution function (the integral of the 
	PDF between -inf and x) calculated by brute-force. **This function 
	should be avoided** as it is very slow, only here for testing 
	purposes.
	"""
	ct.check_are_instances({'landau_x_mpv':landau_x_mpv, 'landau_xi':landau_xi, 'gauss_sigma':gauss_sigma}, (int, float))
	ct.check_is_instance(x, 'x', (int, float, np.ndarray))
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
	
	Error handling
	--------------
	Rises `TypeError` if the parameters are not within the accepted types.
	Non valid values (e.g. sigma<0) rise no errors but return `float('NaN')`.
	
	Notes
	-----
	For certain values of the parameters an approximation is returned. This
	is to avoid numerical instabilities and to increase the calculation
	speed. Such approximations are:
	- If `gauss_sigma < 1e-6*landau_xi` it returns a Landau.
	- If `gauss_sigma > 10000*landau_xi` it returns a Gaussian.
	"""
	ct.check_are_instances({'landau_x_mpv':landau_x_mpv, 'landau_xi':landau_xi, 'gauss_sigma':gauss_sigma, 'lower_n_xi_sigma':lower_n_xi_sigma, 'dx_n_xi':dx_n_xi}, (int, float))
	ct.check_is_instance(x, 'x', (int, float, np.ndarray))
	if isinstance(x, (int, float)):
		x = np.array([x])
	if gauss_sigma < 0 or landau_xi < 0 or gauss_sigma == landau_xi == 0: # Non valid values.
		result = x*float('NaN')
	else: # sigma>=0 and xi>=0 but sigma != xi
		if gauss_sigma < 1e-6*landau_xi: # The Gaussian contribution is negligible...
			result = landau.cdf(x, landau_x_mpv, landau_xi)
		elif gauss_sigma > 10000*landau_xi: # The Landau contribution is negligible...
			result = (erf((x-landau_x_mpv)/gauss_sigma)+1)/2
		else: # All cases where Landau and Gauss contributions are important simultaneously...
			x_low = np.minimum(x, landau_x_mpv - lower_n_xi_sigma*(landau_xi+gauss_sigma)) # At this point the PDF is 1e-9 smaller than in the peak, and goes very quickly to 0.
			x_high = x
			dx = landau_xi/dx_n_xi
			xx = np.linspace(x_low, x_high, int(max(x_high-x_low)/dx))
			xx[xx>x_high] = float('NaN')
			result = np.trapz(
				x = xx,
				y = pdf(xx.reshape(xx.shape[0]*xx.shape[1]),landau_x_mpv, landau_xi, gauss_sigma).reshape(xx.shape),
				axis = 0,
			)
	return np.squeeze(result)

def sample(landau_x_mpv: float, landau_xi: float, gauss_sigma: float, n_samples: int):
	"""Generate samples from a langauss distribution.
	
	Parameters
	----------
	landau_x_mpv: float
		The most probable value of the Landau component of the langauss
		distribution from which to generate the samples.
	landau_xi: float
		The width of the Landau component of the langauss distribution
		from which to generate the samples.
	gauss_sigma: float
		The standard deviation of the Gaussian component of the langauss
		distribution from which to generate the samples.
	n_samples: int
		The number of samples to generate.
	
	Returns
	-------
	samples: float, numpy array
		The samples from the langauss distribution.
	"""
	ct.check_are_instances({'landau_x_mpv':landau_x_mpv, 'landau_xi':landau_xi, 'gauss_sigma':gauss_sigma}, (int, float))
	ct.check_is_instance(n_samples, 'n_samples', int)
	if landau_xi == gauss_sigma == 0:
		raise ValueError(f'`landau_xi` and `gauss_sigma` are both 0, the langauss distribution is not defined there so I cannot sample it.')
	if landau_xi < 0:
		raise ValueError(f'`landau_xi` must be >= 0.')
	if gauss_sigma < 0:
		raise ValueError(f'`gauss_sigma` must be >= 0.')
	if n_samples < 0:
		raise ValueError(f'`n_samples` must be > 0.')
	x_axis = np.linspace(landau_x_mpv - 6*(landau_xi+gauss_sigma),landau_x_mpv+(55*landau_xi+6*gauss_sigma),333)
	return sample_distribution_given_cdf(x_axis, cdf(x_axis,landau_x_mpv,landau_xi,gauss_sigma), n_samples)
