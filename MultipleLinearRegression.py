import numpy as np

class MultipleLinearRegression:
    def __init__(self): #I would need intercept and slopes(coefficients, weights)
        self.intercept=None 
        self.slopes=None
    def fit(self, X,Y): #Method fit should take in inputs as paramaters
        ### calculate intercept and slopes
        # first, calculate the slope using a formula taking X, y
        X_mean=np.mean(X,axis=0)
        Y_mean=np.mean(Y)
        Result=[]    
        # initiate the numbers as zeros before adding in the loop
        self.slopes=np.zeros(features)#how many heatures, how many slopes  
        for j in range(features): #col
            
            for i in range(obs):  #row
                numerator=0
                denominator=0
                numerator += (X[i][j] - X_mean[j])*(Y[i] - Y_mean) # 7*2 --> 2*7 ** 7*1 ==2*1
                denominator += ((X[i][j] - X_mean[j])*(X[i][j] - X_mean[j]))
                ratio=numerator/denominator
            Result.append(ratio)
        self.slopes=np.matrix(Result).T
        self.intercept=Y_mean-X_mean*self.slopes # 1*2 **  1*2---->2*1
        
    
    
    def predict(self,X):    
   
            y_pred=self.intercept+X*self.slopes # 4*2 2*1
            return y_pred
    
model=MultipleLinearRegression()
model.fit(X,Y)

print(model.predict(INP))
