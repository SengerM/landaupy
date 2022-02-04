import unittest
import numpy as np
from landaupy import landau, langauss
from math import isclose

DEBUGGING_PLOTS = True

if DEBUGGING_PLOTS == True:
	import plotly.graph_objects as go

def areclose(A, B, rel_tol=1e-09, abs_tol=0.0) -> bool:
	"""An extension of `math.isclose` to arrays."""
	return all(isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) for a,b in zip(A,B))

class TestLandauValues(unittest.TestCase):
	"""Test if the functions are producing the correct values."""
	def test_pdf(self):
		x_mpv_to_test = [0,1,11,111,1111,1e-3,1e-11]
		x_mpv_to_test += [-x for x in x_mpv_to_test]
		xi_to_test = [1e-22,1e-3,1,11,111,1e5,1e22]
		
		if DEBUGGING_PLOTS == True:
			fig = go.Figure()
			fig.update_layout(title='Landau PDF')
		
		for x_mpv in x_mpv_to_test:
			for xi in xi_to_test:
				with self.subTest(i={'x_mpv': x_mpv, 'xi': xi}):
					x = np.linspace(x_mpv-5*xi, x_mpv+22*xi,999)
					pdf_by_landaupy = landau.pdf(x, x_mpv, xi)
					pdf_reference = np.array([landau.pdf_not_vectorized(x, x_mpv, xi) for x in x])
					
					if DEBUGGING_PLOTS == True:
						if 'legend_group_number' not in locals():
							legend_group_number = 0
						fig.add_trace(go.Scatter(x=x, y=pdf_by_landaupy, name=f'landaupy x_mpv={x_mpv} xi={xi}', legendgroup=f'{legend_group_number}'))
						fig.add_trace(go.Scatter(x=x, y=pdf_reference, name=f'reference x_mpv={x_mpv} xi={xi}', legendgroup=f'{legend_group_number}'))
						legend_group_number += 1
					
					self.assertTrue(
						areclose(
							A = pdf_by_landaupy,
							B = pdf_reference,
							rel_tol = 1e-99,
						)
					)
		
		if DEBUGGING_PLOTS == True:
			fig.show()

class TestLangaussValues(unittest.TestCase):
	"""Test if the functions are producing the correct values."""
	def test_pdf(self):
		x_mpv_to_test = [0]
		xi_to_test = [.01,1]
		sigma_to_test = xi_to_test
		
		if DEBUGGING_PLOTS == True:
			fig = go.Figure()
			fig.update_layout(title='Langauss PDF')
		
		for x_mpv in x_mpv_to_test:
			for xi in xi_to_test:
				for sigma in sigma_to_test:
					with self.subTest(i={'x_mpv': x_mpv, 'xi': xi, 'sigma': sigma}):
						x = np.linspace(x_mpv-5*xi, x_mpv+22*xi,9)
						pdf_by_landaupy = langauss.pdf(x, x_mpv, xi, sigma)
						pdf_reference = np.array([langauss.pdf_not_vectorized(x, x_mpv, xi, sigma) for x in x])
						
						if DEBUGGING_PLOTS == True:
							if 'legend_group_number' not in locals():
								legend_group_number = 0
							fig.add_trace(go.Scatter(x=x, y=pdf_by_landaupy, name=f'landaupy x_mpv={x_mpv} xi={xi} sigma={sigma}', legendgroup=f'{legend_group_number}'))
							fig.add_trace(go.Scatter(x=x, y=pdf_reference, name=f'reference x_mpv={x_mpv} xi={xi} sigma={sigma}', legendgroup=f'{legend_group_number}'))
							legend_group_number += 1
						
						self.assertTrue(
							areclose(
								A = pdf_by_landaupy,
								B = pdf_reference,
								rel_tol = 1e-3,
							)
						)
		
		if DEBUGGING_PLOTS == True:
			fig.show()

if __name__ == '__main__':
	unittest.main()
