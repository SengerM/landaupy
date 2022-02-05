"""This script implements tests for the most fundamental functions of
this package.
"""

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
		self.assertTrue(all(landau.landau_pdf(np.array(9*[-float('inf')]))==0))
		self.assertTrue(all(landau.landau_pdf(np.array(9*[float('inf')]))==0))
	
	def test_with_NaN(self):
		self.assertTrue(np.isnan(landau.landau_pdf(float('NaN'))))
		self.assertTrue(all(np.isnan(landau.landau_pdf(np.array([float('NaN')]*999)))))
		self.assertTrue(any(np.isnan(landau.landau_pdf(np.array([float('NaN')]+[1,2,3,4,5])))))
	
	def test_with_finite_numbers(self):
		self.assertFalse(np.isnan(landau.landau_pdf(4)))
		self.assertFalse(any(np.isnan(landau.landau_pdf(np.linspace(-22,2222,9999999)))))
		self.assertFalse(any(np.isnan(landau.landau_pdf(np.linspace(-1e99,1e99,9999999)))))
	
	def test_normalization(self):
		integral, err = integrate.quad(landau.landau_pdf, -float('inf'), float('inf'))
		self.assertTrue(
			isclose(
				a = integral,
				b = 1,
				rel_tol = 1e-9,
			),
		)
	
	def test_not_rises_error(self):
		for x in [1,1.1,float('inf'),-float('inf'),float('NaN'),np.array(1),np.array([1,2]),np.random.random((5,6))]:
			with self.subTest(i={'x': x}):
				try:
					landau.landau_pdf(x)
				except:
					self.fail()
	
	def test_rises_type_error(self):
		for x in ['a',[1,2,3]]:
			with self.subTest(i={'x': x}):
				with self.assertRaises(TypeError):
					landau.landau_pdf(x)
	
	def test_shape(self):
		# Test floats and ints ---
		for x in [1,1.1,float('NaN'),float('inf')]:
			with self.subTest(i=f'x={x} of type {type(x)}'):
				result = landau.landau_pdf(x)
				self.assertTrue(
					isinstance(result, float),
					f'Was expecting `result` of type `{float}` but instead if is of type `{type(result)}`.'
				)
		# Test numpy arrays ---
		for x in [np.array(0), np.linspace(1,2), np.random.random((3)),np.random.random((3,3)),np.random.random((3,3,3)),np.random.random((3,3,3,3))]:
			with self.subTest(i=f'`x` of type {type(x)} with shape {x.shape}'):
				result = landau.landau_pdf(x)
				self.assertTrue(
					isinstance(result, np.ndarray),
					f'Was expecting `result` of type `{np.ndarray}` but instead it is of type `{type(result)}`.'
				)
				self.assertTrue(
					result.shape == x.shape,
					f'Was expecting `result.shape` to be `{x.shape}` but instead it is `{result.shape}`.'
				)
	
	def test_against_reference(self):
		x = np.linspace(-999, 9999,99999)
		pdf_by_landaupy = landau.landau_pdf(x)
		pdf_reference = np.array([landau.pdf_not_vectorized(x, landau.convert_x0_to_x_mpv(0,1), 1) for x in x])
		
		self.assertTrue(
			np.allclose(
				pdf_by_landaupy,
				pdf_reference,
				equal_nan = True,
				rtol = 1e-9,
				atol = 1e-9,
			)
		)

class TestLandauCDF(unittest.TestCase):
	"""Tests of the `landaupy.landau.landau_cdf` function."""
	
	def test_at_infinities(self):
		self.assertEqual(landau.landau_cdf(-float('inf')), 0)
		self.assertEqual(landau.landau_cdf(+float('inf')), 1)
	
	def test_with_NaN(self):
		self.assertTrue(np.isnan(landau.landau_cdf(float('NaN'))))
		self.assertTrue(all(np.isnan(landau.landau_cdf(np.array([float('NaN')]*999)))))
		self.assertTrue(any(np.isnan(landau.landau_cdf(np.array([float('NaN')]+[1,2,3,4,5])))))
	
	def test_with_finite_numbers(self):
		self.assertFalse(any(np.isnan(landau.landau_pdf(np.linspace(-22,2222,999)))))
		self.assertFalse(any(np.isnan(landau.landau_pdf(np.linspace(-1e99,1e99,999)))))
	
	def test_not_rises_error(self):
		for x in [1,1.1,float('inf'),-float('inf'),float('NaN'),np.array(1),np.array([1,2]),np.random.random((5,6))]:
			with self.subTest(i={'x': x}):
				try:
					landau.landau_cdf(x)
				except:
					self.fail()
	
	def test_rises_type_error(self):
		for x in ['a',[1,2,3]]:
			with self.subTest(i={'x': x}):
				with self.assertRaises(TypeError):
					landau.landau_cdf(x)
	
	def test_shape(self):
		# Test floats and ints ---
		for x in [1,1.1,float('NaN'),float('inf')]:
			with self.subTest(i=f'x={x} of type {type(x)}'):
				result = landau.landau_cdf(x)
				self.assertTrue(
					isinstance(result, float),
					f'Was expecting `result` of type `{float}` but instead if is of type `{type(result)}`.'
				)
		# Test numpy arrays ---
		for x in [np.array(0), np.linspace(1,2), np.random.random((3)),np.random.random((3,3)),np.random.random((3,3,3)),np.random.random((3,3,3,3))]:
			with self.subTest(i=f'`x` of type {type(x)} with shape {x.shape}'):
				result = landau.landau_cdf(x)
				self.assertTrue(
					isinstance(result, np.ndarray),
					f'Was expecting `result` of type `{np.ndarray}` but instead it is of type `{type(result)}`.'
				)
				self.assertTrue(
					result.shape == x.shape,
					f'Was expecting `result.shape` to be `{x.shape}` but instead it is `{result.shape}`.'
				)

if __name__ == '__main__':
	unittest.main()
