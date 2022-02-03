# landaupy

A simple and **pure Python implementation**¹ of [the Landau distribution](https://en.wikipedia.org/wiki/Landau_distribution), since no common package (Scipy, Numpy, etc.) provides this. The algorithm to calculate the Landau distribution was adapted from [the Root implementation](https://root.cern.ch/doc/master/PdfFuncMathCore_8cxx_source.html) which is [documented here](https://root.cern.ch/doc/master/group__PdfFunc.html#ga53d01e04de833eda26560c40eb207cab).

## Installation

```
pip install landaupy
```

## Usage

See [the examples](examples).

### Landau distribution

A simple example of the Landau distribution:
```python
import landaupy.landau as landau
import numpy as np

pdf = landau.pdf(x=1,x_mpv=2,xi=3) # Calculate in a single point.
print(pdf)

pdf = landau.pdf(
	x = np.linspace(-11,111,9999), # Calculate along a numpy array.
	x_mpv = 2, # `x_mpv` is also vectorized, you can put a numpy array here.
	xi = 3 # `xi` is also vectorized, you can put a numpy array here.
)
print(pdf)
```
The Landau cumulative distribution function is also available:
```python
import landaupy.landau as landau
import numpy as np

cdf = landau.cdf(x=1,x_mpv=2,xi=3) # Calculate in a single point.
print(cdf)

cdf = landau.cdf(
	x = np.linspace(-11,111,999), # Calculate along a numpy array.
	x_mpv = 2,
	xi = 3
)
print(pdf)
```
For more, see [the examples](examples).

### Langauss distribution

I also implemented the so called **langauss** distribution which is the convolution of a Gaussian and a Landau, useful when working with real life particle detectors. The usage is similar:
A simple example of the Landau distribution:
```python
import landaupy.langauss as langauss
import numpy as np

pdf = langauss.pdf(x=1,landau_x_mpv=2,landau_xi=3,gauss_sigma=3) # Calculate in a single point.
print(pdf)

pdf = langauss.pdf(
	x = np.linspace(-11,111,999), # Calculate along a numpy array.
	landau_x_mpv = 2,
	landau_xi = 3,
	gauss_sigma = 3
)
print(pdf)
```
The Landau cumulative distribution function is also available:
```python
import landaupy.langauss as langauss
import numpy as np

cdf = langauss.cdf(x=1,landau_x_mpv=2,landau_xi=3,gauss_sigma=3) # Calculate in a single point.
print(cdf)

cdf = langauss.cdf(
	x = np.linspace(-11,111,99), # Calculate along a numpy array.
	landau_x_mpv = 2,
	landau_xi = 3,
	gauss_sigma = 3
)
print(pdf)
```
For more, see [the examples](examples).

## Differences WRT the Root version

Despite this implementation is based in the original from Root, I made a few changes to be more consistent with the rest of the world.

### Normalization

One of the things I changed is the normalization of the `langauss.pdf` distribution such that it integrates to 1. **All the PDFs in this package integrate to 1**. You can verify the normalization with the following code:
```python
import landaupy.landau as landau
import landaupy.langauss as langauss
import numpy as np
import scipy.integrate as integrate
import warnings

with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	# Silence warnings since the numerical integrations between -inf and +inf are prompt to show some warnings.
	for x_mpv in [0,10,55]:
		for xi in [1,2,10]:
			# Integrate to verify normalization (check also how fast it does the calculation):
			print(f'x_mpv={x_mpv}, xi={xi} → integral(landau.pdf) = {integrate.quad(lambda x: landau.pdf(x, x_mpv=x_mpv, xi=xi), -float("inf"), float("inf"))[0]}')
			for sigma in [.1,1,5,10]:
				print(f'x_mpv={x_mpv}, xi={xi}, sigma={sigma} → integral(langauss.pdf) = {integrate.quad(lambda x: langauss.pdf(x, landau_x_mpv=x_mpv, landau_xi=xi, gauss_sigma=sigma), -float("inf"), float("inf"))[0]}')
```

### The MPV argument

One of the arguments of the `landau.pdf` function is `x_mpv`. This is really the *x* position of the Most Probable Value (MPV), which is usually what one is interested in. In the original implementation from Root the parameter is called `x0` and it is close to the MPV but it is not the MPV. The relation between them is given by `x0 = x_mpv + 0.22278298*xi` and I stole it from [the Root implementation of the `langauss` function](https://root.cern.ch/doc/master/langaus_8C.html).

## Footnotes

¹ Only extra requirement is [numpy](https://numpy.org/), which is trivial to install in my experience, and not any strange thing like C/C++→Python compilers/bridges, etc. that will fail or be hard to install.
