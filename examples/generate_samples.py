import numpy as np
from landaupy import landau, langauss
import plotly.graph_objects as go

# Landau samples ---

fig = go.Figure()
for x_mpv, xi in zip([1,20,30],[1,4,2]):
		samples = landau.sample(x_mpv, xi, 9999)

		fig.add_trace(
			go.Histogram(
				x = samples, 
				histnorm = 'probability density',
				name = f'{len(samples)} samples, x<sub>MPV</sub>={x_mpv}, ξ={xi}',
				legendgroup = f'{x_mpv}{xi}',
			)
		)
		x_axis = np.linspace(min(samples),max(samples),999)
		fig.add_trace(
			go.Scatter(
				x = x_axis,
				y = landau.pdf(x_axis, x_mpv, xi),
				name = f'PDF, x<sub>MPV</sub>={x_mpv}, ξ={xi}',
				legendgroup = f'{x_mpv}{xi}',
			)
		)
fig.update_layout(
	xaxis_title = 'x',
	yaxis_title = 'PDF',
	title = 'Sampling a Landau distribution<br><sup>Created in Python with landaupy</sup>',
)
fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
fig.show()

# Langauss samples ---

fig = go.Figure()
for x_mpv, xi, sigma in zip([1,20,30],[1,4,2],[1,3,9]):
		samples = langauss.sample(x_mpv, xi, sigma, 9999)

		fig.add_trace(
			go.Histogram(
				x = samples, 
				histnorm = 'probability density',
				name = f'{len(samples)} samples, x<sub>MPV</sub>={x_mpv}, ξ={xi}, σ={sigma}',
				legendgroup = f'{x_mpv}{xi}',
			)
		)
		x_axis = np.linspace(min(samples),max(samples),999)
		fig.add_trace(
			go.Scatter(
				x = x_axis,
				y = langauss.pdf(x_axis, x_mpv, xi, sigma),
				name = f'PDF, x<sub>MPV</sub>={x_mpv}, ξ={xi}, σ={sigma}',
				legendgroup = f'{x_mpv}{xi}',
			)
		)
fig.update_layout(
	xaxis_title = 'x',
	yaxis_title = 'PDF',
	title = 'Sampling a langauss distribution<br><sup>Created in Python with landaupy</sup>',
)
fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
fig.show()
