import numpy as np
from landaupy import langauss
from scipy.stats import median_abs_deviation
from scipy.optimize import curve_fit
import plotly.graph_objects as go

def binned_fit_langauss(samples, bins='auto', nan='remove'):
	if nan == 'remove':
		samples = samples[~np.isnan(samples)]
	hist, bin_edges = np.histogram(samples, bins, density=True)
	bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2
	# Add an extra bin to the left:
	hist = np.insert(hist, 0, sum(samples<bin_edges[0]))
	bin_centers = np.insert(bin_centers, 0, bin_centers[0]-np.diff(bin_edges)[0])
	# Add an extra bin to the right:
	hist = np.append(hist,sum(samples>bin_edges[-1]))
	bin_centers = np.append(bin_centers, bin_centers[-1]+np.diff(bin_edges)[0])
	landau_x_mpv_guess = bin_centers[np.argmax(hist)]
	landau_xi_guess = median_abs_deviation(samples)/5
	gauss_sigma_guess = landau_xi_guess/10
	popt, pcov = curve_fit(
		lambda x, mpv, xi, sigma: langauss.pdf(x, mpv, xi, sigma),
		xdata = bin_centers,
		ydata = hist,
		p0 = [landau_x_mpv_guess, landau_xi_guess, gauss_sigma_guess],
	)
	return popt, pcov, hist, bin_centers

LANDAU_X_MPV = 5
LANDAU_XI = 1
GAUSS_SIGMA = 2

samples = langauss.sample(landau_x_mpv = LANDAU_X_MPV, landau_xi = LANDAU_XI, gauss_sigma = GAUSS_SIGMA, n_samples = 222)
popt, _, hist, bin_centers = binned_fit_langauss(samples)

fig = go.Figure()
fig.update_layout(
	xaxis_title = 'x',
	yaxis_title = 'PDF',
	title = 'Langauss fit<br><sup>Created in Python with landaupy</sup>',
)
fig.add_trace(
	go.Histogram(
		x = samples,
		name = f'<b>Langauss samples</b><br>x<sub>MPV</sub>={LANDAU_X_MPV:.2e}<br>ξ={LANDAU_XI:.2e}<br>σ={GAUSS_SIGMA:.2e}',
		histnorm = 'probability density',
		nbinsx = 55,
	)
)
x_axis = np.linspace(min(samples),max(samples),999)
fig.add_trace(
	go.Scatter(
		x = x_axis,
		y = langauss.pdf(x_axis, *popt),
		name = f'<b>Langauss fit</b><br>x<sub>MPV</sub>={popt[0]:.2e}<br>ξ={popt[1]:.2e}<br>σ={popt[2]:.2e}',
	)
)
fig.show()
