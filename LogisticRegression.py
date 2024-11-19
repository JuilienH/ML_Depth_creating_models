
# Logistic regression has four basic attributes: learning rate, the number of iterations, weights(coef) and bias(residuals)
# In the model code: usually two methods: fit, predict
import numpy as np
class LogisticRegression:

    def __init__(self,learning_rate=100, n_iters=10):  # These two attributes have starting values. so next two lines can reference to them on the right hand side.
        self.learning_rate=learning_rate
        self.n_iters = n_iters
        self.weights = None #no starting value
        self.bias =None # no starting value
    def fit(self, X, y):
        # initialize the wieghts and bias
        samples,features=X.shape # the number of row is sample number, the number of columns are the number of features
        self.weights= np.zeros(features)
        self.bias = 0
        for i in range(self.n_iters):
            # predicted values should be either 0 or 1, following sigmoid function the formula provided 
            z=np.dot(X,self.weights)+self.bias  # the fornula weights*inputs+bias
            y_pred=1/(1+np.exp(-z)) # feed the calculated probabilities in sigmoid function

            # the log loss function provided for logistic regression:
            #loss = (-1 / samples) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            # calculate gradients provided for weights and bias
            dw = (1 / samples) * np.dot(X.T, (y_pred - y)) # this is tricky to remember each row contains multiple features, but Transpose group  all rows byx1, x2,
            db = (1 / samples) * np.sum(y_pred - y)

            # penalize weights and bias learning rate* gradients
            self.weights -=self.learning_rate*dw
            self.bias -=self.learning_rate*db
    def predict(self, X):
        z=np.dot(X,self.weights)+self.bias  # the fornula weights*inputs+bias
        y_pred=1/(1+np.exp(-z))
        return np.round(y_pred).astype(int)

    ##################################################################################################

    # create sample dataset
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1, 1])

# initialize logistic regression model
lr = LogisticRegression()

# train model on sample dataset
lr.fit(X, y)

# make predictions on new data
X_new = np.array([[6, 7], [7, 8]])
y_pred = lr.predict(X_new)

print(y_pred,lr.weights,lr.bias)  # [1, 1]

        


            
