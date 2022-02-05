import unittest
import numpy as np
from landaupy import landau
import scipy.integrate as integrate
from math import isclose

class TestLandauPDF(unittest.TestCase):
	"""Tests of the `landaupy.landau.landau_pdf` function, which is the most
	fundamental one of the whole package."""
	def test_at_infinities(self):
		self.assertEqual(landau.landau_pdf(-float('inf')), 0)
		self.assertEqual(landau.landau_pdf(+float('inf')), 0)
	
	def test_with_NaN(self):
		self.assertTrue(np.isnan(landau.landau_pdf(float('NaN'))))
		self.assertTrue(all(np.isnan(landau.landau_pdf(np.array([float('NaN')]*999)))))
		self.assertTrue(any(np.isnan(landau.landau_pdf(np.array([float('NaN')]+[1,2,3,4,5])))))
	
	def test_with_finite_numbers(self):
		self.assertFalse(any(np.isnan(landau.landau_pdf(np.linspace(-22,2222,9999999)))))
		self.assertFalse(any(np.isnan(landau.landau_pdf(np.linspace(-1e99,1e99,9999999)))))
	
	def test_rises_type_error(self):
		for x in ['a',[1,2,3]]:
			with self.subTest(i={'x': x}):
				with self.assertRaises(TypeError):
					landau.landau_pdf(x)
	
	def test_normalization(self):
		integral, err = integrate.quad(landau.landau_pdf, -float('inf'), float('inf'))
		self.assertTrue(
			isclose(
				a = integral,
				b = 1,
				rel_tol = 1e-9,
			),
		)
	
if __name__ == '__main__':
	unittest.main()
