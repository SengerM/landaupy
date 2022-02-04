import unittest
import numpy as np
from landaupy import landau
from math import isclose

DEBUGGING_PLOTS = False

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
		
		for x_mpv in x_mpv_to_test:
			for xi in xi_to_test:
				with self.subTest(i={'x_mpv': x_mpv, 'xi': xi}):
					x = np.linspace(x_mpv-5*xi, x_mpv+22*xi,999)
					landau_pdf_by_landaupy = landau.pdf(x, x_mpv, xi)
					landau_pdf_reference = np.array([landau.pdf_not_vectorized(x, x_mpv, xi) for x in x])
					
					if DEBUGGING_PLOTS == True:
						if 'legend_group_number' not in locals():
							legend_group_number = 0
						fig.add_trace(go.Scatter(x=x, y=landau_pdf_by_landaupy, name=f'landaupy x_mpv={x_mpv} xi={xi}', legendgroup=f'{legend_group_number}'))
						fig.add_trace(go.Scatter(x=x, y=landau_pdf_reference, name=f'reference x_mpv={x_mpv} xi={xi}', legendgroup=f'{legend_group_number}'))
						legend_group_number += 1
					
					self.assertTrue(
						areclose(
							A = landau_pdf_by_landaupy,
							B = landau_pdf_reference,
							rel_tol = 1e-99,
						)
					)
		
		if DEBUGGING_PLOTS == True:
			fig.show()

if __name__ == '__main__':
	unittest.main()
