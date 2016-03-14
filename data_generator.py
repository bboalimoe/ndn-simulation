from random import gauss
from theano import function, scan
from theano.tensor import iscalar, fscalar, ivector, fvector, imatrix, fmatrix

class data_generator():
	
	_noise_mu = 0
	_noise_sigma = 1


	def __init__(self, len):
		pass

	def _noise(self):
		return gauss(self._noise_mu, self._noise_sigma)

	def uniform_function(self, K, noise_factor):
		x = ivector('x')
		y = (x/x) * K * (1 + self._noise() * noise_factor)
		f = scan([x], y)
		return f

	def power_function(self, K, noise_factor):
		x = iscalar('x')

if __name__ == "__main__":
	dg = data_generator(0)
	uf = dg.uniform_function(5, 0.01)
	print uf([1, 2, 3, 4])
