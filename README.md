# landaupy

A simple and **pure Python implementation**¹ of [the Landau distribution](https://en.wikipedia.org/wiki/Landau_distribution), since no common package (Scipy, Numpy, ...) provides this. The algorithm to calculate the Landau distribution was adapted from [the Root implementation](https://root.cern.ch/doc/master/PdfFuncMathCore_8cxx_source.html) which is [documented here](https://root.cern.ch/doc/master/group__PdfFunc.html#ga53d01e04de833eda26560c40eb207cab).

## Installation

Clone this repo in `/wherever/you/like` and then run
```
pip install -e /wherever/you/like/landaupy
```

## Usage

A simple example of the **Landau distribution**:

```python
import landaupy.landau as landau
import plotly.graph_objects as go
import numpy as np

print(landau.pdf(x=1,x_mpv=2,xi=3)) # Calculate in a single point.

# Calculate along a Numpy array with fixed parameters:
x_axis = np.linspace(-11,111,9999) # Since it is a "full numpy implementation" it is really fast, even for very large arrays like this one.
fig = go.Figure()
for xi in [1,2]:
	for x_mpv in [0,55]:
		fig.add_trace(
			go.Scatter(
				x = x_axis,
				y = landau.pdf(x_axis, x_mpv=x_mpv, xi=xi),
				name = f'ξ={xi}, x<sub>MPV</sub>={x_mpv}',
			)
		)
fig.update_layout(
	xaxis_title = 'x',
	yaxis_title = 'Landau PDF',
)
fig.show()

# Calculate along any of the parameters:
x_axis = np.linspace(-11,111,999)
xi_values = np.linspace(1,4)
fig = go.Figure()
fig.add_trace(
	go.Heatmap(
		z = landau.pdf(x=x_axis, xi=xi_values, x_mpv=2),
		x = x_axis,
		y = xi_values,
		colorbar = {"title": 'Landau PDF'}
	)
)
fig.update_layout(
	xaxis_title = 'ξ',
	yaxis_title = 'x',
)
fig.show()
```

It was also implemented the **langauss** distribution which is the convolution of a Gaussian and a Landau, useful when working with real life particle detectors. The usage is similar:
```
import landaupy.langauss as langauss
import landaupy.landau as landau
import plotly.graph_objects as go
import numpy as np

print(langauss.pdf(5, 0, 1, 2)) # Calculate the function in a single point.

x_axis = np.linspace(-33,55,999)
fig = go.Figure()
for x_mpv in [0,22]:
	for xi in [1,3]:
		for sigma in [1,5]:
			fig.add_trace(
				go.Scatter(
					x = x_axis,
					y = langauss.pdf(x=x_axis, landau_x_mpv=x_mpv, landau_xi=xi, gauss_sigma=sigma), # Calculate the function very fast over a numpy array.
					name = f'Langauss x<sub>MPV</sub>={x_mpv}, ξ={xi}, σ={sigma}',
				)
			)
		fig.add_trace(
			go.Scatter(
				x = x_axis,
				y = landau.pdf(x_axis, x_mpv, xi),
				name = f'Landau x<sub>MPV</sub>={x_mpv}, ξ={xi}',
			)
		)
fig.update_layout(
	xaxis_title = 'x',
	yaxis_title = 'PDF',
)
fig.show()
```

## Normalization

Despite this implementation is based in the original from Root, I made a few changes to be more consistent with the rest of the world. One of the things I changed is the normalization of the `langauss.pdf` distribution such that it integrates to 1. **All the PDFs in this package integrate to 1**. You can verify the normalization with the following code:
```python
import landaupy.landau as landau
import landaupy.langauss as langauss
import numpy as np
import scipy.integrate as integrate
import warnings

with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	# Silence warnings since the numerical integrations between -inf and +inf are prompt to show some warnings.
	for x_mpv in [0,10,55]:
		for xi in [1,2,10]:
			# Integrate to verify normalization (check also how fast it does the calculation):
			print(f'x_mpv={x_mpv}, xi={xi} → integral(landau.pdf) = {integrate.quad(lambda x: landau.pdf(x, x_mpv=x_mpv, xi=xi), -float("inf"), float("inf"))[0]}')
			for sigma in [.1,1,5,10]:
				print(f'x_mpv={x_mpv}, xi={xi}, sigma={sigma} → integral(langauss.pdf) = {integrate.quad(lambda x: langauss.pdf(x, landau_x_mpv=x_mpv, landau_xi=xi, gauss_sigma=sigma), -float("inf"), float("inf"))[0]}')
```

## Footnotes

¹ Only extra requirement is [numpy](https://numpy.org/), which is trivial to install in my experience, and not any strange thing like C/C++→Python compilers/bridges, etc. that will fail or be hard to install.
