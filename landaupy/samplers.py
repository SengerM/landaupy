import numpy as np
from scipy.interpolate import interp1d

def sample_distribution_given_cdf(x: np.array, cdf: np.array, n_samples: int):
	"""Generate random samples following a particular distribution given
	the CDF.

	Parameters
	----------
	x: numpy array
		Points where the CDF was calculated. `len(x)` must be the same 
		than `len(cdf)`.
	cdf: numpy array
		Value of the CDF at each `x`. `len(x)` must be the same than
		`len(cdf)`.
	n_samples: int
		Number of samples to produce.

	Returns
	-------
	samples: float, numpy array
		Samples produced from the distribution.
	
	Example
	-------
	>>> x_mpv = 0
	>>> xi = 20
	>>> x_axis = np.linspace(x_mpv-5*xi,x_mpv+55*xi,999) # We must be sure to cover the relevant range where the probability is concentrated.
	>>> samples = sample_distribution_given_cdf(x_axis, landau.cdf(x_axis,x_mpv,xi), 99)
	"""
	if any(not isinstance(arg, np.ndarray) for arg in [x, cdf]):
		raise TypeError(f'`x` and `cdf` must be numpy arrays.')
	if not isinstance(n_samples, int):
		raise TypeError(f'`n_samples` must be an integer number.')
	if len(x) != len(cdf):
		raise ValueError(f'`len(x)` must be equal to `len(cdf)`.')
	if n_samples <= 0:
		raise ValueError(f'`n_samples` must be >= 0.')
	inverse_cdf = interp1d(x=cdf, y=x, bounds_error=False)
	samples = np.array([])
	while len(samples)<n_samples:
		samples = np.append(samples, inverse_cdf(np.random.rand(n_samples)))
		samples = samples[~np.isnan(samples)]
	return np.squeeze(samples[:n_samples])
