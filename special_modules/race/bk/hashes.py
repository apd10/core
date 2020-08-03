import numpy as np
from scipy.stats import norm # for P_L2
import math 
from scipy.special import ndtr

class L2LSH():
	def __init__(self, N, d, r): 
		# N = number of hashes 
		# d = dimensionality
		# r = "bandwidth"
		self.N = N
		self.d = d
		self.r = r

		# set up the gaussian random projection vectors
		self.W = np.random.normal(size = (N,d))
		self.b = np.random.uniform(low = 0,high = r,size = N)


	def hash(self,x): 
		return np.floor( (np.squeeze(np.dot(self.W,x)) + self.b)/self.r )


def P_L2(c,w):
	try: 
		p = 1 - 2*ndtr(-w/c) - 2.0/(np.sqrt(2*math.pi)*(w/c)) * (1 - np.exp(-0.5 * (w**2)/(c**2)))
		return p
	except:
		return 1
	# to fix nans, p[np.where(c == 0)] = 1

def P_SRP(x,y): 
	x_u = x / np.linalg.norm(x)
	y_u = y / np.linalg.norm(y)
	angle = np.arccos(np.clip(np.dot(x_u, y_u), -1.0, 1.0))
	return 1.0 - angle / np.pi

class FastSRPMulti():
	# multiple SRP hashes combined into a set of N hash codes
	def __init__(self, reps, d, p): 
		# reps = number of hashes (reps)
		# d = dimensionality
		# p = "bandwidth" = number of hashes (projections) per hash code
		self.N = reps*p # number of hash computations
		self.N_codes = reps # number of output codes
		self.d = d
		self.p = p

		# set up the gaussian random projection vectors
		self.W = np.random.normal(size = (self.N,d))
		self.powersOfTwo = np.array([2**i for i in range(self.p)])

	def hash(self,x): 
		# p is the number of concatenated hashes that go into each
		# of the final output hashes
		h = np.sign( np.dot(self.W,x) )
		h = np.clip( h, 0, 1)
		h = np.reshape(h,(self.N_codes,self.p))
		return np.dot(h,self.powersOfTwo)
