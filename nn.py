import numpy as np
from scipy import optimize

class neuralNet(object):
	def __init__(self):
		self.inputLayerSize = 21
		self.outputLayerSize = 7
		self.hiddenLayerSize = 14
		self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)

	def forward(self,X):
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yHat = self.sigmoid(self.z3)
		return yHat

	def sigmoid(self,dataVector):
		return 1/(1+np.exp(-dataVector))
	def sigmoidDerivate(self,dataVector):
		return np.exp(-dataVector)/((1+np.exp(-dataVector))**2)
	def costFunction(self, X, y):
		self.yHat = self.forward(X)
		J = 0.5*sum((y-self.yHat)**2)
		return J
	def costFunctionPrime(self, X, y):
		self.yHat = self.forward(X)

		delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
		dJdW2 = np.dot(self.a2.T, delta3)

		delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
		dJdW1 = np.dot(X.T, delta2)

		return dJdW1, dJdW2

	def getParams(self):
		params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
		return params
	def setParams(self, params):
		W1_start = 0
		W1_end = self.hiddenLayerSize * self.inputLayerSize
		self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
		W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
		self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

	def computeGradients(self, X, y):
		dJdW1, dJdW2 = self.costFunctionPrime(X, y)
		return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

	def computeNumericalGradient(N, X, y):
		paramsInitial = N.getParams()
		numgrad = np.zeros(paramsInitial.shape)
		perturb = np.zeros(paramsInitial.shape)
		e = 1e-4

		for p in range(len(paramsInitial)):
			perturb[p] = e
			N.setParams(paramsInitial + perturb)
			loss2 = N.costFunction(X, y)

			N.setParams(paramsInitial - perturb)
			loss1 = N.costFunction(X, y)
			numgrad[p] = (loss2 - loss1) / (2*e)
			perturb[p] = 0

		N.setParams(paramsInitial)

		return numgrad


class trainer(object):
	def __init__(self, N):
		self.N = N



	def callbackF(self, params):
		self.N.setParams(params)
		self.J.append(self.N.costFunction(self.X, self.y))

	def costFunctionWrapper(self, params, X, y):
		self.N.setParams(params)
		cost = self.N.costFunction(X, y)
		grad = self.N.computeGradients(X,y)
		return cost, grad

	def train(self, X, y):
		self.X = X
		self.y = y

		self.J = []
		params0 = self.N.getParams()
		options = {'maxiter': 200, 'disp' : True}
		_res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', args=(X, y), options=options, callback=self.callbackF)

		self.N.setParams(_res.x)
		self.optimizationResults = _res


if __name__ == "__main__":
	print "In the function"