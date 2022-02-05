import unittest
import numpy as np
from landaupy import landau

class TestLandauLandau(unittest.TestCase):
	"""Tests of the `landaupy.landau.landau_function` function, which is the most
	fundamental one of the whole package."""
	def test_at_infinities(self):
		self.assertEqual(landau.landau_function(-float('inf')), 0)
		self.assertEqual(landau.landau_function(+float('inf')), 0)
	
	def test_with_NaN(self):
		self.assertTrue(np.isnan(landau.landau_function(float('NaN'))))
		self.assertTrue(all(np.isnan(landau.landau_function(np.array([float('NaN')]*999)))))
		self.assertTrue(any(np.isnan(landau.landau_function(np.array([float('NaN')]+[1,2,3,4,5])))))
	
	def test_with_finite_numbers(self):
		self.assertFalse(any(np.isnan(landau.landau_function(np.linspace(-22,2222,9999999)))))
		self.assertFalse(any(np.isnan(landau.landau_function(np.linspace(-1e99,1e99,9999999)))))
	
	def test_rises_type_error(self):
		for x in ['a',[1,2,3]]:
			with self.subTest(i={'x': x}):
				with self.assertRaises(TypeError):
					landau.landau_function(x)
	
if __name__ == '__main__':
	unittest.main()
