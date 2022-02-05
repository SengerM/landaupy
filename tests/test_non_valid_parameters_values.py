import unittest
import numpy as np
from landaupy import landau, langauss
from math import isclose
import scipy.integrate as integrate

class TestLandauNonValidParamters(unittest.TestCase):
	"""Test that the functions behave as expected when parameters take
	non valid values."""
	def test_pdf(self):
		x_mpv_to_test = [0]
		xi_to_test = [-1,-1e-20,0,1e-20,1]
		
		for x_mpv in x_mpv_to_test:
			for xi in xi_to_test:
				with self.subTest(i={'x_mpv': x_mpv, 'xi': xi}):
					x = np.linspace(x_mpv-55555*np.abs(xi), x_mpv+222222*np.abs(xi),99999)
					pdf_values = landau.pdf(x, x_mpv, xi)
					if xi > 0: # There should be not any NaN
						self.assertFalse(any(np.isnan(pdf_values)), '`landau.pdf` is returning NaN when it should not.')
					else: # All elements in the output should be NaN
						self.assertTrue(all(np.isnan(pdf_values)), '`landau.pdf` is not returning NaN when it should.')
	
	def test_cdf(self):
		x_mpv_to_test = [0]
		xi_to_test = [-1,-1e-20,0,1e-20,1]
		
		for x_mpv in x_mpv_to_test:
			for xi in xi_to_test:
				with self.subTest(i={'x_mpv': x_mpv, 'xi': xi}):
					x = np.linspace(x_mpv-55*np.abs(xi), x_mpv+2222*np.abs(xi),999)
					cdf_values = landau.cdf(x, x_mpv, xi)
					cdf_values[0] = 1
					if xi > 0: # There should be not any NaN
						self.assertFalse(any(np.isnan(cdf_values)), '`landau.cdf` is returning NaN when it should not.')
					else: # All elements in the output should be NaN
						self.assertTrue(all(np.isnan(cdf_values)), '`landau.cdf` is not returning NaN when it should.')

if __name__ == '__main__':
	unittest.main()
