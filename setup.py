import setuptools

setuptools.setup(
	name = "landaupy",
	version = "0.1",
	author = "Matias Senger",
	author_email = "matias.senger@cern.ch",
	description = "A pure python implementation of the Landau distribution",
	url = "https://github.com/SengerM/landaupy",
	packages = setuptools.find_packages(),
	classifiers = [
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
	install_requires = ['numpy'],
	license = 'MIT',
)
