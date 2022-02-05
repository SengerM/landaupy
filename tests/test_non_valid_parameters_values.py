import unittest
import numpy as np
from landaupy import landau, langauss

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
					if xi > 0: # There should be not any NaN
						self.assertFalse(any(np.isnan(cdf_values)), '`landau.cdf` is returning NaN when it should not.')
					else: # All elements in the output should be NaN
						self.assertTrue(all(np.isnan(cdf_values)), '`landau.cdf` is not returning NaN when it should.')

class TestLangaussNonValidParamters(unittest.TestCase):
	"""Test that the functions behave as expected when parameters take
	non valid values."""
	def test_pdf(self):
		x_mpv_to_test = [0]
		xi_to_test = [-1,-1e-20,0,1e-20,1]
		sigma_to_test = [-1,-1e-20,0,1e-20,1]
		
		for x_mpv in x_mpv_to_test:
			for xi in xi_to_test:
				for sigma in sigma_to_test:
					with self.subTest(i={'x_mpv': x_mpv, 'xi': xi, 'sigma': sigma}):
						x = np.linspace(x_mpv-22*(np.abs(xi)+np.abs(sigma)), x_mpv+222*np.abs(xi)+11*np.abs(sigma),9999)
						pdf_values = langauss.pdf(x, x_mpv, xi, sigma)
						should_we_expect_NaN = xi < 0 or sigma < 0 or xi==sigma==0
						if not should_we_expect_NaN:
							self.assertFalse(any(np.isnan(pdf_values)), '`langauss.pdf` is returning NaN when it should not.')
						else: # All elements in the output should be NaN
							self.assertTrue(all(np.isnan(pdf_values)), '`langauss.pdf` is not returning NaN when it should.')
	
	def test_cdf(self):
		x_mpv_to_test = [0]
		xi_to_test = [-1,-1e-20,0,1e-20,1]
		sigma_to_test = [-1,-1e-20,0,1e-20,1]
		
		for x_mpv in x_mpv_to_test:
			for xi in xi_to_test:
				for sigma in sigma_to_test:
					with self.subTest(i={'x_mpv': x_mpv, 'xi': xi, 'sigma': sigma}):
						x = np.linspace(x_mpv-5*(np.abs(xi)+np.abs(sigma)), x_mpv+22*np.abs(xi)+5*np.abs(sigma),99)
						cdf_values = langauss.cdf(x, x_mpv, xi, sigma)
						should_we_expect_NaN = xi < 0 or sigma < 0 or xi==sigma==0
						if not should_we_expect_NaN:
							self.assertFalse(any(np.isnan(cdf_values)), '`langauss.cdf` is returning NaN when it should not.')
						else: # All elements in the output should be NaN
							self.assertTrue(all(np.isnan(cdf_values)), '`langauss.cdf` is not returning NaN when it should.')
	
if __name__ == '__main__':
	unittest.main()
