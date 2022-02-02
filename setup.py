import setuptools

setuptools.setup(
	name = "landaupy",
	version = "0.0",
	author = "Matias Senger",
	author_email = "m.senger@hotmail.com",
	description = "A pure python implementation of the Landau distribution",
	url = "https://github.com/landaupy",
	download_url = "https://github.com/SengerM/landaupy/archive/refs/tags/v0.0.tar.gz",
	packages = setuptools.find_packages(),
	classifiers = [
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
	install_requires = ['numpy'],
)
