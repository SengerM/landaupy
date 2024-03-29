import unittest
import numpy as np
from landaupy import landau, langauss
from math import isclose
import scipy.integrate as integrate

DEBUGGING_PLOTS = False

if DEBUGGING_PLOTS == True:
	import plotly.graph_objects as go
	from plotly.subplots import make_subplots

def areclose(A, B, rel_tol=1e-09, abs_tol=0.0) -> bool:
	"""An extension of `math.isclose` to arrays."""
	return all(isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) for a,b in zip(A,B))

class TestLandauValues(unittest.TestCase):
	"""Test if the functions are producing the correct values."""
	def test_pdf(self):
		"""Compares the `landaupy.pdf` implementation against "the original
		implementation" from Root."""
		x_mpv_to_test = [0,1,11,111,1111,1e-3,1e-11]
		x_mpv_to_test += [-x for x in x_mpv_to_test]
		xi_to_test = [1e-22,1e-3,1,11,111,1e5,1e22]
		
		if DEBUGGING_PLOTS == True:
			fig = make_subplots(rows=2, cols=1)
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
						fig.add_trace(go.Scatter(x=x, y=pdf_by_landaupy, name=f'landaupy x_mpv={x_mpv} xi={xi}', legendgroup=f'{legend_group_number}'), row=1, col=1)
						fig.add_trace(go.Scatter(x=x, y=pdf_reference, name=f'reference x_mpv={x_mpv} xi={xi}', legendgroup=f'{legend_group_number}'), row=1, col=1)
						fig.add_trace(go.Scatter(x=x, y=(pdf_by_landaupy-pdf_reference)/pdf_by_landaupy, name=f'(landaupy-reference)/landaupy', legendgroup=f'{legend_group_number}'), row=2, col=1)
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
	
	def test_normalization(self):
		"""Test that the PDF integrates to 1."""
		x_mpv_to_test = [0,1]
		x_mpv_to_test += [-x for x in x_mpv_to_test]
		xi_to_test = [1e-9,1e-3,1,11,111,1e5,1e22]
		for x_mpv in x_mpv_to_test:
			for xi in xi_to_test:
				with self.subTest(i={'x_mpv': x_mpv, 'xi': xi}):
					integral, err = integrate.quad(lambda x: landau.pdf(x, x_mpv=x_mpv, xi=xi), x_mpv-5*xi, x_mpv+2222*xi)
					self.assertTrue(
						isclose(
							a = integral,
							b = 1,
							abs_tol = 1e-3,
						),
						f'integral={integral}, error={err}, was expecting integral≈1.',
					)
	
	def test_cdf(self):
		"""Test the CDF that starts at 0 and ends up in 1."""
		x_mpv_to_test = [0,1,11,111,1111,1e-3,1e-11]
		x_mpv_to_test += [-x for x in x_mpv_to_test]
		xi_to_test = [1e-12,1e-3,1,11,111,1e5,1e22,1e55] # Very small values here fail because then it tries to evaluate the function in `x_mpv-xi=1-1e-55` wich results in `1` and this is exactly `x_mpv` so the CDF does not return 0 or 1 as one should expect. But this is a problem of the test, not the CDF.
		
		for x_mpv in x_mpv_to_test:
			for xi in xi_to_test:
				with self.subTest(i={'x_mpv': x_mpv, 'xi': xi}):
					x = x_mpv-5*xi
					cdf_value = landau.cdf(x, x_mpv, xi)
					self.assertTrue(
						isclose(
							a = cdf_value,
							b = 0,
							abs_tol = 1e-3,
						),
						f'`landau.cdf` was evaluated in a "very negative point" (x={x}) where it should return "almost 0" but it returned {cdf_value}',
					)
					x = x_mpv+2222*xi
					cdf_value = landau.cdf(x, x_mpv, xi)
					self.assertTrue(
						isclose(
							a = cdf_value,
							b = 1,
							abs_tol = 1e-3,
						),
						f'`landau.cdf` was evaluated in a "very positive point" (x={x}) where it should return "almost 1" but it returned {cdf_value}',
					)
		
class TestLangaussValues(unittest.TestCase):
	"""Test if the functions are producing the correct values."""
	def test_pdf(self):
		"""Compares the `landaupy.pdf` implementation against "the original
		implementation" from Root."""
		x_mpv_to_test = [0]
		xi_to_test = [0] + [10**i for i in [-3,-1,0,1,3]]
		sigma_to_test = xi_to_test
		
		if DEBUGGING_PLOTS == True:
			fig = make_subplots(rows=2, cols=1)
			fig.update_layout(title='Langauss PDF')
		
		for x_mpv in x_mpv_to_test:
			for xi in xi_to_test:
				for sigma in sigma_to_test:
					# TODO: Find a reference for these cases ---
					if sigma >= 1e6*xi or sigma==0:
						continue # The reference function fails here.
					# ------------------------------------------
					if sigma==xi==0: # This is actually a non valid case, but I am not testing that here.
						continue
					with self.subTest(i={'x_mpv': x_mpv, 'xi': xi, 'sigma': sigma}):
						x = np.linspace(x_mpv-2*(xi+sigma), x_mpv+22*xi+5*sigma,9)
						pdf_by_landaupy = langauss.pdf(x, x_mpv, xi, sigma)
						pdf_reference = np.array([langauss.pdf_not_vectorized(x, x_mpv, xi, sigma) for x in x])
						
						if DEBUGGING_PLOTS == True:
							if 'legend_group_number' not in locals():
								legend_group_number = 0
							fig.add_trace(go.Scatter(x=x, y=pdf_by_landaupy, name=f'landaupy x_mpv={x_mpv} xi={xi} sigma={sigma}', legendgroup=f'{legend_group_number}'), row=1, col=1)
							fig.add_trace(go.Scatter(x=x, y=pdf_reference, name=f'reference x_mpv={x_mpv} xi={xi} sigma={sigma}', legendgroup=f'{legend_group_number}'), row=1, col=1)
							fig.add_trace(go.Scatter(x=x, y=(pdf_by_landaupy-pdf_reference)/pdf_by_landaupy, name=f'(landaupy-reference)/landaupy', legendgroup=f'{legend_group_number}'), row=2, col=1)
							legend_group_number += 1
						
						self.assertTrue(
							areclose(
								A = pdf_by_landaupy,
								B = pdf_reference,
								rel_tol = 1e-3,
							),
							'`landaupy.langauss.pdf` does not match the reference.',
						)
		
		if DEBUGGING_PLOTS == True:
			fig.show()
	
	def test_normalization(self):
		"""Test that the PDF integrates to 1."""
		x_mpv_to_test = [0]
		xi_to_test = [0] + [10**i for i in [-9,-6,0,3,9]]
		sigma_to_test = xi_to_test
		for x_mpv in x_mpv_to_test:
			for xi in xi_to_test:
				for sigma in sigma_to_test:
					if sigma==xi==0: # This is actually a non valid case, but I am not testing that here.
						continue
					with self.subTest(i={'x_mpv': x_mpv, 'xi': xi, 'sigma': sigma}):
						integral, err = integrate.quad(lambda x: langauss.pdf(x, landau_x_mpv=x_mpv, landau_xi=xi, gauss_sigma=sigma), x_mpv-5*(xi+sigma), x_mpv+2222*xi+5*sigma)
						self.assertTrue(
							isclose(
								a = integral,
								b = 1,
								abs_tol = 1e-3,
							),
							f'integral={integral}, error={err}, was expecting integral≈1.',
						)
	
	def test_cdf(self):
		"""Test the CDF that starts at 0 and ends up in 1."""
		x_mpv_to_test = [0]
		x_mpv_to_test += [-x for x in x_mpv_to_test]
		xi_to_test = [0] + [10**i for i in [-9,-3,0,3,6,9]] # Very small values here fail because then it tries to evaluate the function in `x_mpv-xi=1-1e-55` wich results in `1` and this is exactly `x_mpv` so the CDF does not return 0 or 1 as one should expect. But this is a problem of the test, not the CDF.
		sigma_to_test = xi_to_test
		
		for x_mpv in x_mpv_to_test:
			for xi in xi_to_test:
				for sigma in sigma_to_test:
					if sigma==xi==0: # This is actually a non valid case, but I am not testing that here.
						continue
					########################################################
					########################################################
					# TODO: fix these values that drain all the memory  ####
					if not 1e-3 < sigma < 1e3 or not 1e-3 < xi < 1e3:   ####
						continue                                        ####
					########################################################
					########################################################
					with self.subTest(i={'x_mpv': x_mpv, 'xi': xi, 'sigma': sigma}):
						x = x_mpv-5*(xi+sigma)
						cdf_value = langauss.cdf(x, x_mpv, xi, sigma)
						self.assertTrue(
							isclose(
								a = cdf_value,
								b = 0,
								abs_tol = 1e-3,
							),
							f'`langauss.cdf` was evaluated in a "very negative point" (x={x}) where it should return "almost 0" but it returned {cdf_value}',
						)
						x = x_mpv+2222*xi+5*sigma
						cdf_value = langauss.cdf(x, x_mpv, xi, sigma)
						self.assertTrue(
							isclose(
								a = cdf_value,
								b = 1,
								abs_tol = 1e-3,
							),
							f'`langauss.cdf` was evaluated in a "very positive point" (x={x}) where it should return "almost 1" but it returned {cdf_value}',
						)

if __name__ == '__main__':
	unittest.main()
