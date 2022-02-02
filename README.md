# landaupy

A simple and **pure Python implementation**¹ of [the Landau distribution](https://en.wikipedia.org/wiki/Landau_distribution), since no common package (Scipy, Numpy, ...) provides this. The algorithm to calculate the Landau distribution was adapted from [the Root implementation](https://root.cern.ch/doc/master/PdfFuncMathCore_8cxx_source.html) which is [documented here](https://root.cern.ch/doc/master/group__PdfFunc.html#ga53d01e04de833eda26560c40eb207cab).

## Installation

Clone this repo in `/wherever/you/like` and then run
```
pip install -e /wherever/you/like/landaupy
```

## Usage

A simple example:

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

## Footnotes

¹ Only extra requirement is [numpy](https://numpy.org/), which is trivial to install in my experience, and not any strange thing like C/C++→Python compilers/bridges, etc. that will fail or be hard to install.
