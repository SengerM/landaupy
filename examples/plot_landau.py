from landaupy import landau
import plotly.graph_objects as go
import numpy as np

XI_VALUES = [.1,1,11]
X_MPV_VALUES = [0,88]

# Let's start with the PDF ---

x_axis = np.linspace(-55,222,9999) # Don't be afraid of using a very big numpy array, the landau.pdf can handle it!

fig = go.Figure()
for xi in XI_VALUES:
	for x_mpv in X_MPV_VALUES:
		fig.add_trace(
			go.Scatter(
				x = x_axis,
				y = landau.pdf(x_axis, x_mpv, xi),
				name = f'x<sub>MPV</sub>={x_mpv}, ξ={xi}',
			)
		)
fig.update_layout(
	xaxis_title = 'x',
	yaxis_title = 'PDF',
	title = 'Landau distribution<br><sup>Created in Python with landaupy</sup>',
)
fig.show()

# Now the CDF ---

x_axis = np.linspace(-55,222,99) # For the CDF it is better to use less points as the integration otherwise drains up a lot of memory.

fig = go.Figure()
for xi in XI_VALUES:
	for x_mpv in X_MPV_VALUES:
		fig.add_trace(
			go.Scatter(
				x = x_axis,
				y = landau.cdf(x_axis, x_mpv, xi),
				name = f'x<sub>MPV</sub>={x_mpv}, ξ={xi}',
			)
		)
fig.update_layout(
	xaxis_title = 'x',
	yaxis_title = 'CDF',
	title = 'Landau cumulative probability function<br><sup>Created in Python with landaupy</sup>',
)
fig.show()
