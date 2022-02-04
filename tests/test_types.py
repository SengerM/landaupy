import unittest
import numpy as np
from landaupy import landau, langauss

class TestLandauTypes(unittest.TestCase):
	
	def test_pdf(self):
		landau_pdf_should_not_rise_type_error = [
			{'x': 1, 'x_mpv': 1, 'xi': 1},
			{'x': 1.1, 'x_mpv': 1.2, 'xi': 1.3},
			{'x': np.linspace(1,2), 'x_mpv': 1, 'xi': 1},
			{'x': 1, 'x_mpv': np.linspace(1,2), 'xi': 1},
			{'x': 1, 'x_mpv': 1, 'xi': np.linspace(1,2)},
			{'x': np.linspace(1,2), 'x_mpv': np.linspace(1,2), 'xi': np.linspace(1,2)},
		]
		for args in landau_pdf_should_not_rise_type_error:
			with self.subTest(i=args):
				try:
					landau.pdf(**args)
				except TypeError:
					self.fail()
		
		landau_pdf_shoud_rise_type_error = [
			{'x': '1', 'x_mpv': 1, 'xi': 1},
			{'x': 1, 'x_mpv': '1', 'xi': 1},
			{'x': 1, 'x_mpv': 1, 'xi': '1'},
		]
		for args in landau_pdf_shoud_rise_type_error:
			with self.subTest(i=args):
				with self.assertRaises(TypeError):
					landau.pdf(**args)
		
	def test_cdf(self):
		landau_cdf_should_not_rise_type_error = [
			{'x': 1, 'x_mpv': 1, 'xi': 1, 'lower_n_xi': 4, 'dx_n_xi': 9},
			{'x': np.linspace(1,2), 'x_mpv': 1, 'xi': 1, 'lower_n_xi': 4, 'dx_n_xi': 9},
		]
		for args in landau_cdf_should_not_rise_type_error:
			with self.subTest(i=args):
				try:
					landau.cdf(**args)
				except TypeError:
					self.fail()
		
		landau_cdf_shoud_rise_type_error = [
			{'x': '1', 'x_mpv': 1, 'xi': 1, 'lower_n_xi': 4, 'dx_n_xi': 9},
			{'x': 1, 'x_mpv': '1', 'xi': 1, 'lower_n_xi': 4, 'dx_n_xi': 9},
			{'x': 1, 'x_mpv': np.array(3), 'xi': 1, 'lower_n_xi': 4, 'dx_n_xi': 9},
			{'x': 1, 'x_mpv': 1, 'xi': 1, 'lower_n_xi': 4, 'dx_n_xi': '0'},
		]
		for args in landau_cdf_shoud_rise_type_error:
			with self.subTest(i=args):
				with self.assertRaises(TypeError):
					landau.cdf(**args)
	
	def test_sample(self):
		landau_sample_should_not_rise_type_error = [
			{'x_mpv': 1, 'xi': 1, 'n_samples': 9},
			{'x_mpv': 1.4, 'xi': 1.3, 'n_samples': 9},
		]
		for args in landau_sample_should_not_rise_type_error:
			with self.subTest(i=args):
				try:
					landau.sample(**args)
				except TypeError:
					self.fail()
		
		landau_sample_shoud_rise_type_error = [
			{'x_mpv': np.linspace(1,2), 'xi': 1, 'n_samples': 9},
			{'x_mpv': 1, 'xi': np.linspace(1,2), 'n_samples': 9},
			{'x_mpv': 1, 'xi': 1, 'n_samples': np.linspace(1,2)},
			{'x_mpv': 1, 'xi': 1, 'n_samples': 9.4},
		]
		for args in landau_sample_shoud_rise_type_error:
			with self.subTest(i=args):
				with self.assertRaises(TypeError):
					landau.sample(**args)

class TestLangaussTypes(unittest.TestCase):
	
	def test_pdf(self):
		langauss_pdf_should_not_rise_type_error = [
			{'x': 1, 'landau_x_mpv': 1, 'landau_xi': 1, 'gauss_sigma': 1},
			{'x': 1.1, 'landau_x_mpv': 2.1, 'landau_xi': 3.1, 'gauss_sigma': 4.1},
			{'x': np.linspace(1,2), 'landau_x_mpv': 1, 'landau_xi': 1, 'gauss_sigma': 1},
		]
		for args in langauss_pdf_should_not_rise_type_error:
			with self.subTest(i=args):
				try:
					langauss.pdf(**args)
				except TypeError:
					self.fail()
		
		langauss_pdf_shoud_rise_type_error = [
			{'x': '1', 'landau_x_mpv': 1, 'landau_xi': 1, 'gauss_sigma': 1},
			{'x': 1, 'landau_x_mpv': np.linspace(1,2), 'landau_xi': 1, 'gauss_sigma': 1},
			{'x': 1, 'landau_x_mpv': 1, 'landau_xi': np.linspace(1,2), 'gauss_sigma': 1},
			{'x': 1, 'landau_x_mpv': 1, 'landau_xi': 1, 'gauss_sigma': np.linspace(1,2)},
		]
		for args in langauss_pdf_shoud_rise_type_error:
			with self.subTest(i=args):
				with self.assertRaises(TypeError):
					langauss.pdf(**args)
		
	def test_cdf(self):
		langauss_cdf_should_not_rise_type_error = [
			{'x': 1, 'landau_x_mpv': 1, 'landau_xi': 1, 'gauss_sigma': 1, 'lower_n_xi_sigma': 4, 'dx_n_xi': 4},
			{'x': np.linspace(1,2), 'landau_x_mpv': 1, 'landau_xi': 1, 'gauss_sigma': 1, 'lower_n_xi_sigma': 4, 'dx_n_xi': 4},
		]
		for args in langauss_cdf_should_not_rise_type_error:
			with self.subTest(i=args):
				try:
					langauss.cdf(**args)
				except TypeError:
					self.fail()
		
		langauss_cdf_shoud_rise_type_error = [
			{'x': '1', 'landau_x_mpv': 1, 'landau_xi': 1, 'gauss_sigma': 1, 'lower_n_xi_sigma': 4, 'dx_n_xi': 4},
			{'x': 1, 'landau_x_mpv': np.linspace(1,2), 'landau_xi': 1, 'gauss_sigma': 1, 'lower_n_xi_sigma': 4, 'dx_n_xi': 4},
			{'x': 1, 'landau_x_mpv': 1, 'landau_xi': np.linspace(1,2), 'gauss_sigma': 1, 'lower_n_xi_sigma': 4, 'dx_n_xi': 4},
			{'x': 1, 'landau_x_mpv': 1, 'landau_xi': 1, 'gauss_sigma': np.linspace(1,2), 'lower_n_xi_sigma': 4, 'dx_n_xi': 4},
			{'x': 1, 'landau_x_mpv': 1, 'landau_xi': 1, 'gauss_sigma': 1, 'lower_n_xi_sigma': np.linspace(1,2), 'dx_n_xi': 4},
			{'x': 1, 'landau_x_mpv': 1, 'landau_xi': 1, 'gauss_sigma': 1, 'lower_n_xi_sigma': 4, 'dx_n_xi': np.linspace(1,2)},
		]
		for args in langauss_cdf_shoud_rise_type_error:
			with self.subTest(i=args):
				with self.assertRaises(TypeError):
					langauss.cdf(**args)
	
	def test_sample(self):
		langauss_sample_should_not_rise_type_error = [
			{'landau_x_mpv': 1, 'landau_xi': 1, 'gauss_sigma': 1, 'n_samples': 9},
			# ~ {'landau_x_mpv': 1.2, 'landau_xi': 1.1, 'gauss_sigma': 1.4, 'n_samples': 9},
		]
		for args in langauss_sample_should_not_rise_type_error:
			with self.subTest(i=args):
				try:
					langauss.sample(**args)
				except TypeError:
					self.fail()
		
		langauss_sample_shoud_rise_type_error = [
			{'landau_x_mpv': np.linspace(1,2), 'landau_xi': 1, 'gauss_sigma': 1, 'n_samples': 9},
			{'landau_x_mpv': 1, 'landau_xi': np.linspace(1,2), 'gauss_sigma': 1, 'n_samples': 9},
			{'landau_x_mpv': 1, 'landau_xi': 1, 'gauss_sigma': np.linspace(1,2), 'n_samples': 9},
			{'landau_x_mpv': 1, 'landau_xi': 1, 'gauss_sigma': 1, 'n_samples': np.linspace(1,2)},
			{'landau_x_mpv': 1, 'landau_xi': 1, 'gauss_sigma': 1, 'n_samples': 9.4},
		]
		for args in langauss_sample_shoud_rise_type_error:
			with self.subTest(i=args):
				with self.assertRaises(TypeError):
					langauss.sample(**args)

if __name__ == '__main__':
	unittest.main()
