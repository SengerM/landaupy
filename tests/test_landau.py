"""This script implements tests for the most fundamental functions of
this package.
"""

import unittest
import numpy as np
from landaupy import landau
import scipy.integrate as integrate
from math import isclose
import warnings

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
	
	def test_returns_floats(self):
		# Test floats and ints ---
		for x in [1,1.1,float('NaN'),float('inf')]:
			with self.subTest(i=f'x={x} of type {type(x)}'):
				result = landau.landau_pdf(x)
				self.assertTrue(
					isinstance(result, float),
					f'Was expecting `result` of type `{float}` but instead if is of type `{type(result)}`.'
				)
	
	def test_returns_propper_shape_array(self):
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
		self.assertTrue(all(landau.landau_cdf(np.array(9*[-float('inf')]))==0))
		self.assertTrue(all(landau.landau_cdf(np.array(9*[float('inf')]))==1))
	
	def test_with_NaN(self):
		self.assertTrue(np.isnan(landau.landau_cdf(float('NaN'))))
		self.assertTrue(all(np.isnan(landau.landau_cdf(np.array([float('NaN')]*999)))))
		self.assertTrue(any(np.isnan(landau.landau_cdf(np.array([float('NaN')]+[1,2,3,4,5])))))
	
	def test_with_finite_numbers(self):
		self.assertFalse(np.isnan(landau.landau_cdf(2)))
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
	
	def test_returns_floats(self):
		for x in [1,1.1,float('NaN'),float('inf')]:
			with self.subTest(i=f'x={x} of type {type(x)}'):
				result = landau.landau_cdf(x)
				self.assertTrue(
					isinstance(result, float),
					f'Was expecting `result` of type `{float}` but instead if is of type `{type(result)}`.'
				)
	
	def test_returns_propper_shape_array(self):
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
	
	def test_against_reference(self):
		x = np.linspace(-11, 222,99)
		cdf_by_landaupy = landau.landau_cdf(x)
		cdf_reference = np.array([landau.automatic_cdf(x, landau.convert_x0_to_x_mpv(0,1), 1) for x in x])
		
		self.assertTrue(
			np.allclose(
				cdf_by_landaupy,
				cdf_reference,
				equal_nan = True,
				rtol = 1e-3,
				atol = 1e-3,
			)
		)

class TestPDFWithFloatParameters(unittest.TestCase):
	"""Tests for the `pdf` function of the `landau` module using single 
	float numbers for the parameters `x_mpv` and `xi`.
	"""
	x_mpv_to_test = [0,1,1e3,1e-3,1e-6,1e-9,1e-12,1e9]
	x_mpv_to_test += [-x for x in x_mpv_to_test]
	xi_to_test = [1e-9,1e-3,1,11,111,1e5,1e22]
	
	def test_at_infinities(self):
		for x_mpv in self.x_mpv_to_test:
			for xi in self.xi_to_test:
				with self.subTest(i={'x_mpv': x_mpv, 'xi': xi}):
					self.assertEqual(landau.pdf(-float('inf'), x_mpv, xi), 0)
					self.assertEqual(landau.pdf(+float('inf'), x_mpv, xi), 0)
					self.assertTrue(all(landau.pdf(np.array(9*[-float('inf')]), x_mpv, xi)==0))
					self.assertTrue(all(landau.pdf(np.array(9*[float('inf')]), x_mpv, xi)==0))
	
	def test_with_NaN(self):
		for x_mpv in self.x_mpv_to_test:
			for xi in self.xi_to_test:
				with self.subTest(i={'x_mpv': x_mpv, 'xi': xi}):
					self.assertTrue(np.isnan(landau.pdf(float('NaN'), x_mpv, xi)))
					self.assertTrue(all(np.isnan(landau.pdf(np.array([float('NaN')]*999), x_mpv, xi))))
					self.assertTrue(any(np.isnan(landau.pdf(np.array([float('NaN')]+[1,2,3,4,5]), x_mpv, xi))))
				
	def test_with_finite_numbers(self):
		for x_mpv in self.x_mpv_to_test:
			for xi in self.xi_to_test:
				with self.subTest(i={'x_mpv': x_mpv, 'xi': xi}):
					self.assertFalse(np.isnan(landau.pdf(4, x_mpv, xi)))
					self.assertFalse(any(np.isnan(landau.pdf(np.linspace(x_mpv-5*xi,x_mpv+2222*xi,999), x_mpv, xi))))
	
	def test_normalization(self):
		for x_mpv in self.x_mpv_to_test:
			for xi in self.xi_to_test:
				with self.subTest(i={'x_mpv': x_mpv, 'xi': xi}):
					with warnings.catch_warnings():
						warnings.simplefilter("ignore")
						integral, err = integrate.quad(lambda x: landau.pdf(x, x_mpv, xi), x_mpv-5*xi, x_mpv+2222*xi)
					if err>1: # We cannot do anything...
						self.skipTest(f"Cannot run `{self.__class__.__name__}.{self.test_normalization.__name__}` for parameters { {'x_mpv': x_mpv, 'xi': xi} } because the integration method for testing (scipy.integrate.quad?) is producing an error of {err}.")
					self.assertTrue(
						isclose(
							a = integral,
							b = 1,
							rel_tol = 1e-3,
						),
						f'`landau.pdf` is not integrating to 1, integration result: integral={integral}, error={err}.'
					)
	
	def test_not_rises_error(self):
		for x_mpv in self.x_mpv_to_test:
			for xi in self.xi_to_test:
				for x in [1,1.1,float('inf'),-float('inf'),float('NaN'),np.array(1),np.array([1,2]),np.random.random((5,6))]:
					with self.subTest(i={'x': x, 'x_mpv': x_mpv, 'xi': xi}):
						try:
							landau.pdf(x, x_mpv, xi)
						except:
							self.fail()
	
	def test_rises_type_error(self):
		for x_mpv in self.x_mpv_to_test:
			for xi in self.xi_to_test:
				for x in ['a',[1,2,3]]:
					with self.subTest(i={'x': x, 'x_mpv': x_mpv, 'xi': xi}):
						with self.assertRaises(TypeError):
							landau.pdf(x, x_mpv, xi)
	
	def test_returns_floats(self):
		# Test floats and ints ---
		for x_mpv in self.x_mpv_to_test:
			for xi in self.xi_to_test:
				for x in [1,1.1,float('NaN'),float('inf')]:
					with self.subTest(i=f'x={x} of type {type(x)}'):
						result = landau.pdf(x, x_mpv, xi)
						self.assertTrue(
							isinstance(result, float),
							f'Was expecting `result` of type `{float}` but instead if is of type `{type(result)}`.'
						)
	
	def test_returns_propper_shape_array(self):
		# Test numpy arrays ---
		for x_mpv in self.x_mpv_to_test:
			for xi in self.xi_to_test:
				for x in [np.array(0), np.linspace(1,2), np.random.random((3)),np.random.random((3,3)),np.random.random((3,3,3)),np.random.random((3,3,3,3))]:
					with self.subTest(i=f'`x` of type {type(x)} with shape {x.shape}'):
						result = landau.pdf(x, x_mpv, xi)
						self.assertTrue(
							isinstance(result, np.ndarray),
							f'Was expecting `result` of type `{np.ndarray}` but instead it is of type `{type(result)}`.'
						)
						self.assertTrue(
							result.shape == x.shape,
							f'Was expecting `result.shape` to be `{x.shape}` but instead it is `{result.shape}`.'
						)
	
	def test_against_reference(self):
		for x_mpv in self.x_mpv_to_test:
			for xi in self.xi_to_test:
				x = np.linspace(x_mpv-5*xi,x_mpv+2222*xi,999)
				with self.subTest(i={'x': x, 'x_mpv': x_mpv, 'xi': xi}):
					pdf_by_landaupy = landau.pdf(x, x_mpv, xi)
					pdf_reference = np.array([landau.pdf_not_vectorized(x, x_mpv, xi) for x in x])
					self.assertTrue(
						np.allclose(
							pdf_by_landaupy,
							pdf_reference,
							equal_nan = True,
							rtol = 1e-9,
							atol = 1e-9,
						)
					)


if __name__ == '__main__':
	unittest.main()
