class trainer(object):
    def __init__(self, N):
        #Make local reference to Neural Network:
        self.N = N

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)

    def train(self, X, y):

        params0 = self.N.getParams()

        options = {'maxiter':200, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method'BFGS', args=(X,y), options=options)