# This code was adapted from the Root implementation on 2.Feb.2022 by Matías Senger (matias.senger@cern.ch). The links below point to the Root documentation and implementation.
# https://root.cern.ch/doc/master/group__PdfFunc.html#ga53d01e04de833eda26560c40eb207cab
# https://root.cern.ch/doc/master/PdfFuncMathCore_8cxx_source.html

import numpy as np
from .samplers import sample_distribution_given_cdf
from . import _check_types as ct
import warnings

def pdf_not_vectorized(x: float, x_mpv: float, xi: float) -> float:
	"""Non vectorized Landau PDF calculation. **This function should be 
	avoided**, this is almost a copy-paste from [the original Root code](https://root.cern.ch/doc/master/PdfFuncMathCore_8cxx_source.html)
	only for testing purposes.
	"""
	ct.check_are_instances({'x':x,'x_mpv':x_mpv,'xi':xi}, (int, float))
	
	p1 = (0.4259894875, -0.1249762550, 0.03984243700, -0.006298287635, 0.001511162253)
	q1 = (1.0, -0.3388260629, 0.09594393323, -0.01608042283, 0.003778942063)
	p2 = (0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411, 0.0001283617211)
	q2 = (1.0, 0.7428795082, 0.3153932961, 0.06694219548, 0.008790609714)
	p3 = (0.1788544503, 0.09359161662, 0.006325387654, 0.00006611667319, -0.000002031049101)
	q3 = (1.0, 0.6097809921, 0.2560616665, 0.04746722384, 0.006957301675)
	p4 = (0.9874054407, 118.6723273, 849.2794360, -743.7792444, 427.0262186)
	q4 = (1.0, 106.8615961, 337.6496214, 2016.712389, 1597.063511)
	p5 = (1.003675074, 167.5702434, 4789.711289, 21217.86767, -22324.94910)
	q5 = (1.0, 156.9424537, 3745.310488, 9834.698876, 66924.28357)
	p6 = (1.000827619, 664.9143136, 62972.92665, 475554.6998, -5743609.109)
	q6 = (1.0, 651.4101098, 56974.73333, 165917.4725, -2815759.939)
	a1 = (0.04166666667, -0.01996527778, 0.02709538966)
	a2 = (-1.845568670, -4.284640743)
	
	x_mpv = x_mpv + 0.22278298*xi # This number I took from Root's langauss implementation: https://root.cern.ch/doc/master/langaus_8C.html and basically it gives the correct MPV value.
	
	if xi <= 0:
		return 0
	v = (x - x_mpv) / xi
	if v < -5.5:
		u = np.exp(v + 1.0)
		if u < 1e-10:
			return 0.0
		ue = np.exp(-1 / u)
		us = np.sqrt(u)
		denlan = 0.3989422803 * (ue / us) * (1 + (a1[0] + (a1[1] + a1[2] * u) * u) * u)
	elif v < -1:
		u = np.exp(-v - 1)
		denlan = np.exp(-u) * np.sqrt(u) * (p1[0] + (p1[1] + (p1[2] + (p1[3] + p1[4] * v) * v) * v) * v) / (q1[0] + (q1[1] + (q1[2] + (q1[3] + q1[4] * v) * v) * v) * v)
	elif v < 1:
		denlan = (p2[0] + (p2[1] + (p2[2] + (p2[3] + p2[4] * v) * v) * v) * v) / (q2[0] + (q2[1] + (q2[2] + (q2[3] + q2[4] * v) * v) * v) * v)
	elif v < 5:
		denlan = (p3[0] + (p3[1] + (p3[2] + (p3[3] + p3[4] * v) * v) * v) * v) / (q3[0] + (q3[1] + (q3[2] + (q3[3] + q3[4] * v) * v) * v) * v)
	elif v < 12:
		u = 1 / v
		denlan = u * u * (p4[0] + (p4[1] + (p4[2] + (p4[3] + p4[4] * u) * u) * u) * u) / (q4[0] + (q4[1] + (q4[2] + (q4[3] + q4[4] * u) * u) * u) * u)
	elif v < 50:
		u = 1 / v
		denlan = u * u * (p5[0] + (p5[1] + (p5[2] + (p5[3] + p5[4] * u) * u) * u) * u) / (q5[0] + (q5[1] + (q5[2] + (q5[3] + q5[4] * u) * u) * u) * u)
	elif v < 300:
		u = 1 / v
		denlan = u * u * (p6[0] + (p6[1] + (p6[2] + (p6[3] + p6[4] * u) * u) * u) * u) / (q6[0] + (q6[1] + (q6[2] + (q6[3] + q6[4] * u) * u) * u) * u)
	else:
		u = 1 / (v - v * np.log(v) / (v + 1))
		denlan = u * u * (1 + (a2[0] + a2[1] * u) * u)
	return denlan / xi

def landau_pdf(x):
	"""Calculates the "basic" Landau distribution, i.e. the distribution
	when the location parameter is 0 and the scale parameter is 1. The 
	algorithm was adapted from [the Root implementation](https://root.cern.ch/doc/master/PdfFuncMathCore_8cxx_source.html).
	
	Parameters
	----------
	x: float, numpy array
		Point in which to calculate the function.
	
	Returns
	-------
	landau_pdf: float, numpy array
		The value of the Landau distribution.
	
	Error handling
	--------------
	Rises `TypeError` if the parameters are not within the accepted types.
	"""
	def denlan_1(x):
		"""Calculates denlan when x < -5.5. If x is outside this range, NaN value is returned."""
		a1 = (0.04166666667, -0.01996527778, 0.02709538966)
		u = np.exp(x+1)
		denlan = 0.3989422803*(np.exp(-1/u)/u**.5)*(1 + (a1[0] + (a1[1] + a1[2]*u)*u)*u)
		denlan[u<1e-10] = 0
		denlan[x>=-5.5] = float('NaN')
		return denlan

	def denlan_2(x):
		"""Calculates denlan when -5.5 <= x < -1. If x is outside this range, NaN value is returned."""
		p1 = (0.4259894875, -0.1249762550, 0.03984243700, -0.006298287635, 0.001511162253)
		q1 = (1.0, -0.3388260629, 0.09594393323, -0.01608042283, 0.003778942063)
		u = np.exp(-x-1)
		denlan = np.exp(-u)*np.sqrt(u)*(p1[0] + (p1[1] + (p1[2] + (p1[3] + p1[4]*x)*x)*x)*x)/(q1[0] + (q1[1] + (q1[2] + (q1[3] + q1[4]*x)*x)*x)*x)
		denlan[(x<-5.5)|(x>=-1)] = float('NaN')
		return denlan

	def denlan_3(x):
		"""Calculates denlan when -1 <= x < 1. If x is outside this range, NaN value is returned."""
		p2 = (0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411, 0.0001283617211)
		q2 = (1.0, 0.7428795082, 0.3153932961, 0.06694219548, 0.008790609714)
		denlan = (p2[0] + (p2[1] + (p2[2] + (p2[3] + p2[4]*x)*x)*x)*x)/(q2[0] + (q2[1] + (q2[2] + (q2[3] + q2[4]*x)*x)*x)*x)
		denlan[(x<-1)|(x>=1)] = float('NaN')
		return denlan

	def denlan_4(x):
		"""Calculates denlan when 1 <= x < 5. If x is outside this range, NaN value is returned."""
		p3 = (0.1788544503, 0.09359161662, 0.006325387654, 0.00006611667319, -0.000002031049101)
		q3 = (1.0, 0.6097809921, 0.2560616665, 0.04746722384, 0.006957301675)
		denlan = (p3[0] + (p3[1] + (p3[2] + (p3[3] + p3[4]*x)*x)*x)*x) / (q3[0] + (q3[1] + (q3[2] + (q3[3] + q3[4]*x)*x)*x)*x)
		denlan[(x<1)|(x>=5)] = float('NaN')
		return denlan

	def denlan_5(x):
		"""Calculates denlan when 5 <= x < 12. If x is outside this range, NaN value is returned."""
		p4 = (0.9874054407, 118.6723273, 849.2794360, -743.7792444, 427.0262186)
		q4 = (1.0, 106.8615961, 337.6496214, 2016.712389, 1597.063511)
		u = 1/x
		denlan = u * u * (p4[0] + (p4[1] + (p4[2] + (p4[3] + p4[4] * u) * u) * u) * u) / (q4[0] + (q4[1] + (q4[2] + (q4[3] + q4[4] * u) * u) * u) * u)
		denlan[(x<5)|(x>=12)] = float('NaN')
		return denlan

	def denlan_6(x):
		"""Calculates denlan when 12 <= x < 50. If x is outside this range, NaN value is returned."""
		p5 = (1.003675074, 167.5702434, 4789.711289, 21217.86767, -22324.94910)
		q5 = (1.0, 156.9424537, 3745.310488, 9834.698876, 66924.28357)
		u = 1/x
		denlan = u * u * (p5[0] + (p5[1] + (p5[2] + (p5[3] + p5[4] * u) * u) * u) * u) / (q5[0] + (q5[1] + (q5[2] + (q5[3] + q5[4] * u) * u) * u) * u)
		denlan[(x<12)|(x>=50)] = float('NaN')
		return denlan

	def denlan_7(x):
		"""Calculates denlan when 50 <= x < 300. If x is outside this range, NaN value is returned."""
		p6 = (1.000827619, 664.9143136, 62972.92665, 475554.6998, -5743609.109)
		q6 = (1.0, 651.4101098, 56974.73333, 165917.4725, -2815759.939)
		u = 1 / x
		denlan = u * u * (p6[0] + (p6[1] + (p6[2] + (p6[3] + p6[4] * u) * u) * u) * u) / (q6[0] + (q6[1] + (q6[2] + (q6[3] + q6[4] * u) * u) * u) * u)
		denlan[(x<50)|(x>=300)] = float('NaN')
		return denlan

	def denlan_8(x):
		"""Calculates denlan when x >= 300. If x is outside this range, NaN value is returned."""
		a2 = (-1.845568670, -4.284640743)
		u = 1 / (x - x * np.log(x) / (x + 1))
		denlan = u * u * (1 + (a2[0] + a2[1] * u) * u)
		denlan[x<=300] = float('NaN')
		return denlan
	
	ct.check_is_instance(x, 'x', (int, float, np.ndarray))
	x, = np.meshgrid(x)
	x = x.astype(float)
	
	result = x*float('NaN') # Initialize
	x_is_finite_indices = np.isfinite(x)
	with warnings.catch_warnings():
		warnings.simplefilter("ignore") # I don't want to see the warnings of numpy, anyway it will fill with `NaN` values so it is fine.
		denlan = x[x_is_finite_indices]*float('NaN') # Initialize.
		limits = (-float('inf'),  -5.5,       -1,        1,        5,       12,       50,      300, float('inf'))
		formulas = (denlan_1, denlan_2, denlan_3, denlan_4, denlan_5, denlan_6, denlan_7, denlan_8)
		for k, formula in enumerate(formulas):
			indices = (limits[k]<=x[x_is_finite_indices])&(x[x_is_finite_indices]<limits[k+1])
			denlan[indices] = formula(x[x_is_finite_indices][indices])
		result[x_is_finite_indices] = denlan
	result[np.isinf(x)] = 0
	
	return np.squeeze(result)

def landau_cdf(x, x_min: float=-5, x_max: float=9999, dx: float=1):
	"""Calculates the CDF of the "basic" Landau distribution, i.e. the 
	distribution when the location parameter is 0 and the scale parameter 
	is 1. The algorithm was adapted from [the Root implementation](https://root.cern.ch/doc/master/PdfFuncMathCore_8cxx_source.html).
	
	Parameters
	----------
	x: float, numpy array
		Point in which to calculate the function.
	x_min: float, default -5
		Value of $x$ from which to start the integration. Reduce for more
		precision. Values of `x` below `x_min` will return `0`.
	x_max: float, default 9999
		Value at which stop the integration. Values of `x` above `x_max`
		will return `1`.
	dx: float, default 1
		Step for the integration. Reduce for more precision. Although 1
		may look like a very big number, it should be more than enough 
		for most applications.
	
	Returns
	-------
	landau_cdf: float, numpy array
		The value of the Landau distribution.
	
	Error handling
	--------------
	Rises `TypeError` if the parameters are not within the accepted types.
	If `dx` <= 0 it rises `ValueError`.
	"""
	ct.check_is_instance(x, 'x', (int, float, np.ndarray))
	ct.check_are_instances({'dx': dx, 'x_min': x_min}, (int, float))
	if dx <= 0:
		raise ValueError(f'`dx` must be > 0.')
	if not np.isfinite(x_min):
		raise ValueError(f'`x_min` must be finite, I have received x_min={repr(x_min)}.')
	if not np.isfinite(dx):
		raise ValueError(f'`x_min` must be finite, I have received x_min={repr(dx)}.')
	
	x, = np.meshgrid(x)
	x = x.astype(float)
	
	result = x*float('NaN') # Initialize.
	
	x_is_finite_indices = np.isfinite(x)
	if x_is_finite_indices.any():
		x_low = np.minimum(x[x_is_finite_indices], x_min).astype(float)
		x_high = np.minimum(x[x_is_finite_indices], x_max).astype(float)
		xx = np.linspace(x_low, x_high, int(max(x_high-x_low)/dx))
		xx[xx>x_high] = float('NaN')
		result[x_is_finite_indices] = np.trapz(
			x = xx,
			y = landau_pdf(xx.reshape(xx.shape[0]*xx.shape[1])).reshape(xx.shape),
			axis = 0,
		)
	result[x<x_min] = 0
	result[x>x_max] = 1
	result[np.isneginf(x)] = 0
	result[np.isposinf(x)] = 1
	return np.squeeze(result)

def pdf(x, x_mpv, xi):
	"""Landau probability density function (PDF) with parameters.

	Parameters
	----------
	x: float, numpy array
		Point (or points) where to calculate the PDF.
	x_mpv: float, numpy array
		Position of the most probable value (MPV) of the Landau distribution.
	xi: float, numpy array
		Parameter $xi$ of the Landau distribution, it is a measure of its width.

	Returns
	-------
	landau_pdf: float, numpy array
		Value of the Landau PDF.
	
	Error handling
	--------------
	Rises `TypeError` if the parameters are not within the accepted types.
	Non valid values (e.g. xi<0) rise no errors but return `float('NaN')`.
	"""
	
	ct.check_are_instances({'x':x, 'x_mpv':x_mpv, 'xi':xi}, (int, float, np.ndarray))
	
	x = np.asarray(x).astype(float)
	x_mpv = np.asarray(x_mpv).astype(float)
	xi = np.asarray(xi).astype(float)
	if xi.shape == () and xi <= 0: # If `xi` is just a single number and <= 0...
		result = x*float('NaN')
	else:
		x0 = x_mpv + 0.22278298*xi # This number I took from Root's langauss implementation: https://root.cern.ch/doc/master/langaus_8C.html and basically it gives the correct MPV value.
		with warnings.catch_warnings():
			warnings.simplefilter("ignore") # I don't want to see the warnings of numpy, anyway it will fill with `NaN` or `inf` values so it is fine.
			result = landau_pdf((x - x0) / xi)/xi
		if xi.shape != ():
			result[xi<=0] = float('NaN')
	return np.squeeze(result)

def automatic_cdf(x, x_mpv: float, xi: float):
	"""Landau cumulative distribution function (the integral of the PDF 
	between -inf and x) calculated by brute-force. **This function should 
	be avoided** as it is very slow, only here for testing purposes.
	"""
	ct.check_are_instances({'x_mpv':x_mpv, 'xi':xi}, (int, float))
	ct.check_is_instance(x, 'x', (int, float, np.ndarray))
	from scipy.integrate import quad
	integrand = lambda X: pdf(X, x_mpv, xi)
	def _cdf(x):
		return quad(integrand, -float('inf'), x)
	_cdf = np.vectorize(_cdf)
	integral, error = _cdf(x)
	return integral

def cdf(x, x_mpv: float, xi: float, lower_n_xi: float=4, dx_n_xi: float=9):
	"""Landau cumulative distribution function (CDF). 

	Parameters
	----------
	x: float, numpy array
		Point (or points) where to calculate the CDF.
	x_mpv: float
		Position of the most probable value (MPV) of the Landau distribution.
	xi: float
		Parameter $xi$ of the Landau distribution, it is a measure of its width.
	lower_n_xi: float, default 4
		The numeric integration lower limit is dependent on `xi` in the 
		following way `np.minimum(x, x_mpv - lower_n_xi*xi)`. The default 
		value should work in any case but if you find troubles you can 
		change it. Increasing `lower_n_xi` will extend the integration 
		towards -infinity yielding more accurate results but extending 
		the computation time.
	dx_n_xi: float, default 9
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
	ct.check_are_instances({'x_mpv':x_mpv, 'xi':xi, 'lower_n_xi':lower_n_xi, 'dx_n_xi':dx_n_xi}, (int, float))
	ct.check_is_instance(x, 'x', (int, float, np.ndarray))
	for name, var in {'lower_n_xi': lower_n_xi, 'dx_n_xi': dx_n_xi}.items():
		if var <= 0:
			raise ValueError(f'`{name}` must be > 0.')
	if xi <= 0:
		result = x*float('NaN')
	else: # xi > 0
		if isinstance(x, (int, float)):
			x = np.array([x])
		result = landau_cdf((x - x0) / xi)
	return np.squeeze(result)

def sample(x_mpv: float, xi: float, n_samples: int):
	"""Generate samples from a Landau distribution.
	
	Parameters
	----------
	x_mpv: float
		The most probable value of the Landau distribution from which to 
		generate the samples.
	xi: float
		The width of the Landau distribution from which to generate the
		samples.
	n_samples: int
		The number of samples to generate.
	
	Returns
	-------
	samples: float, numpy array
		The samples from the Landau distribution.
	"""
	ct.check_are_instances({'x_mpv':x_mpv, 'xi':xi}, (int, float))
	ct.check_is_instance(n_samples, 'n_samples', (int))
	if xi <= 0:
		raise ValueError(f'`xi` must be > 0.')
	if n_samples <= 0:
		raise ValueError(f'`n_samples` must be > 0.')
	x_axis = np.linspace(x_mpv-5*xi,x_mpv+55*xi,999)
	return sample_distribution_given_cdf(x_axis, cdf(x_axis,x_mpv,xi), n_samples)
