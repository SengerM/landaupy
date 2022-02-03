# This code was adapted from the Root implementation on 2.Feb.2022 by Matías Senger (matias.senger@cern.ch). The links below point to the Root documentation and implementation.
# https://root.cern.ch/doc/master/group__PdfFunc.html#ga53d01e04de833eda26560c40eb207cab
# https://root.cern.ch/doc/master/PdfFuncMathCore_8cxx_source.html

import numpy as np

def pdf_not_vectorized(x: float, x_mpv: float, xi: float) -> float:
	"""Non vectorized Landau PDF calculation. **This function should be avoided**, this is almost a copy-paste from the original Root code in https://root.cern.ch/doc/master/PdfFuncMathCore_8cxx_source.html only for testing purposes."""
	if any([not isinstance(arg, (int,float)) for arg in [x,x_mpv,xi]]):
		raise TypeError('All arguments must be float numbers.')
	
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

def pdf(x, x_mpv, xi):
	"""Landau probability density function (PDF). The algorithm was
	adapted from [the Root implementation](https://root.cern.ch/doc/master/PdfFuncMathCore_8cxx_source.html).

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
	"""
	def denlan_1(v):
		"""Calculates denlan when v < -5.5. If v is outside this range, NaN value is returned."""
		a1 = (0.04166666667, -0.01996527778, 0.02709538966)
		u = np.exp(v+1)
		denlan = 0.3989422803*(np.exp(-1/u)/u**.5)*(1 + (a1[0] + (a1[1] + a1[2]*u)*u)*u)
		denlan[u<1e-10] = 0
		denlan[v>=-5.5] = float('NaN')
		return denlan

	def denlan_2(v):
		"""Calculates denlan when -5.5 <= v < -1. If v is outside this range, NaN value is returned."""
		p1 = (0.4259894875, -0.1249762550, 0.03984243700, -0.006298287635, 0.001511162253)
		q1 = (1.0, -0.3388260629, 0.09594393323, -0.01608042283, 0.003778942063)
		u = np.exp(-v-1)
		denlan = np.exp(-u)*np.sqrt(u)*(p1[0] + (p1[1] + (p1[2] + (p1[3] + p1[4]*v)*v)*v)*v)/(q1[0] + (q1[1] + (q1[2] + (q1[3] + q1[4]*v)*v)*v)*v)
		denlan[(v<-5.5)|(v>=-1)] = float('NaN')
		return denlan

	def denlan_3(v):
		"""Calculates denlan when -1 <= v < 1. If v is outside this range, NaN value is returned."""
		p2 = (0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411, 0.0001283617211)
		q2 = (1.0, 0.7428795082, 0.3153932961, 0.06694219548, 0.008790609714)
		denlan = (p2[0] + (p2[1] + (p2[2] + (p2[3] + p2[4]*v)*v)*v)*v)/(q2[0] + (q2[1] + (q2[2] + (q2[3] + q2[4]*v)*v)*v)*v)
		denlan[(v<-1)|(v>=1)] = float('NaN')
		return denlan

	def denlan_4(v):
		"""Calculates denlan when 1 <= v < 5. If v is outside this range, NaN value is returned."""
		p3 = (0.1788544503, 0.09359161662, 0.006325387654, 0.00006611667319, -0.000002031049101)
		q3 = (1.0, 0.6097809921, 0.2560616665, 0.04746722384, 0.006957301675)
		denlan = (p3[0] + (p3[1] + (p3[2] + (p3[3] + p3[4]*v)*v)*v)*v) / (q3[0] + (q3[1] + (q3[2] + (q3[3] + q3[4]*v)*v)*v)*v)
		denlan[(v<1)|(v>=5)] = float('NaN')
		return denlan

	def denlan_5(v):
		"""Calculates denlan when 5 <= v < 12. If v is outside this range, NaN value is returned."""
		p4 = (0.9874054407, 118.6723273, 849.2794360, -743.7792444, 427.0262186)
		q4 = (1.0, 106.8615961, 337.6496214, 2016.712389, 1597.063511)
		u = 1/v
		denlan = u * u * (p4[0] + (p4[1] + (p4[2] + (p4[3] + p4[4] * u) * u) * u) * u) / (q4[0] + (q4[1] + (q4[2] + (q4[3] + q4[4] * u) * u) * u) * u)
		denlan[(v<5)|(v>=12)] = float('NaN')
		return denlan

	def denlan_6(v):
		"""Calculates denlan when 12 <= v < 50. If v is outside this range, NaN value is returned."""
		p5 = (1.003675074, 167.5702434, 4789.711289, 21217.86767, -22324.94910)
		q5 = (1.0, 156.9424537, 3745.310488, 9834.698876, 66924.28357)
		u = 1/v
		denlan = u * u * (p5[0] + (p5[1] + (p5[2] + (p5[3] + p5[4] * u) * u) * u) * u) / (q5[0] + (q5[1] + (q5[2] + (q5[3] + q5[4] * u) * u) * u) * u)
		denlan[(v<12)|(v>=50)] = float('NaN')
		return denlan

	def denlan_7(v):
		"""Calculates denlan when 50 <= v < 300. If v is outside this range, NaN value is returned."""
		p6 = (1.000827619, 664.9143136, 62972.92665, 475554.6998, -5743609.109)
		q6 = (1.0, 651.4101098, 56974.73333, 165917.4725, -2815759.939)
		u = 1 / v
		denlan = u * u * (p6[0] + (p6[1] + (p6[2] + (p6[3] + p6[4] * u) * u) * u) * u) / (q6[0] + (q6[1] + (q6[2] + (q6[3] + q6[4] * u) * u) * u) * u)
		denlan[(v<50)|(v>=300)] = float('NaN')
		return denlan

	def denlan_8(v):
		"""Calculates denlan when v >= 300. If v is outside this range, NaN value is returned."""
		a2 = (-1.845568670, -4.284640743)
		u = 1 / (v - v * np.log(v) / (v + 1))
		denlan = u * u * (1 + (a2[0] + a2[1] * u) * u)
		denlan[v<=300] = float('NaN')
		return denlan
	
	x, x_mpv, xi = np.meshgrid(x,x_mpv,xi)
	x_mpv = x_mpv + 0.22278298*xi # This number I took from Root's langauss implementation: https://root.cern.ch/doc/master/langaus_8C.html and basically it gives the correct MPV value.
	v = (x - x_mpv) / xi
	
	denlan = x*float('NaN') # Initialize.
	denlan[xi<=0] = 0
	
	limits = (-float('inf'),  -5.5,       -1,        1,        5,       12,       50,      300, float('inf'))
	formulas = (denlan_1, denlan_2, denlan_3, denlan_4, denlan_5, denlan_6, denlan_7, denlan_8)
	
	for k, formula in enumerate(formulas):
		indices = (limits[k]<=v)&(v<limits[k+1])
		denlan[indices] = formula(v[indices])
	
	return np.squeeze(denlan/xi)

def automatic_cdf(x, x_mpv, xi):
	"""Landau cumulative distribution function (the integral of the PDF between -inf and x) calculated by brute-force. **This function should be avoided** as it is very slow, only here for testing purposes."""
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
	if any([not isinstance(arg, (int,float)) for arg in [x_mpv,xi]]):
		raise TypeError('`x_mpv` and `xi` must be numbers.')
	if not isinstance(x, (int, float, np.ndarray)):
		raise TypeError(f'`x` must be either a number or a numpy array, received object of type {type(x)}.')
	if isinstance(x, (int, float)):
		x = np.array([x])
	x_low = np.minimum(x, x_mpv - lower_n_xi*xi) # At this point the PDF is 1e-9 smaller than in the peak, and goes very quickly to 0.
	x_high = x
	dx = xi/dx_n_xi
	xx = np.linspace(x_low, x_high, int(max(x_high-x_low)/dx))
	xx[xx>x_high] = float('NaN')
	return np.squeeze(
		np.trapz(
			x = xx,
			y = pdf(xx.reshape(xx.shape[0]*xx.shape[1]), x_mpv, xi).reshape(xx.shape),
			axis = 0,
		)
	)
