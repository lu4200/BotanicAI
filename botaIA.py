import numpy as np 

x_entrer = np.array(([3, 1.5], [2, 1], [4, 1.5], [3, 1], [3.5,0.5], [2,0.5], [5.5,1], [1,1], [0.5,1]), dtype=float) #inputs
y = np.array(([1], [0], [1],[0],[1],[0],[1],[0]), dtype=float) # outputs / red = 1 / blue = 0


x_entrer = x_entrer/np.amax(x_entrer, axis=0) # values between 0 and 1 for sigmoidal


X = np.split(x_entrer, [8])[0] # training dataset
xPrediction = np.split(x_entrer, [8])[1] # output waited

#class to set a Neural Network
class Neural_Network(object):
  def __init__(self):
        
  #Arguments
    self.inputSize = 2 # nb neural input
    self.outputSize = 1 # nb neural output
    self.hiddenSize = 3 # nb hidden neurals

  #Weight on branch
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (2x3)
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1)


  #forward propagation
  def forward(self, X):

    self.z = np.dot(X, self.W1) # multiplication M(n,p) input and W1
    self.z2 = self.sigmoid(self.z) # Sigmoidal
    self.z3 = np.dot(self.z2, self.W2) # multiplication M(n,p) hiddenvalues and W2
    o = self.sigmoid(self.z3) # output = result of activation
    return o

  # Sigmoidal
  def sigmoid(self, s):
    return 1/(1+np.exp(-s))

  # Sigprime
  def sigmoidPrime(self, s):
    return s * (1 - s)

  #backwardpropagation
  def backward(self, X, y, o):

    self.o_error = y - o # error between output and and waited value
    self.o_delta = self.o_error*self.sigmoidPrime(o) #sigmoidprime error (x)

    self.z2_error = self.o_delta.dot(self.W2.T) # hidden neurals error 
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # sigmoidprime error on hidden ones

    self.W1 += X.T.dot(self.z2_delta) # adjusting weights with errors
    self.W2 += self.z2.T.dot(self.o_delta) # == W2

  #training 
  def train(self, X, y):
        
    o = self.forward(X)
    self.backward(X, y, o)

  #predicting
  def predict(self):
        
    print("After predicting: ")
    print("Input : \n" + str(xPrediction))
    print("Output : \n" + str(self.forward(xPrediction)))

    if(self.forward(xPrediction) < 0.5):
        print("Blue flower! \n")
    else:
        print("Redm flower! \n")


NN = Neural_Network()

for i in range(13000): #nb of generation careful about overfitting !
    print("# " + str(i) + "\n")
    print("Input values: \n" + str(X))
    print("Actual output: \n" + str(y))
    print("Predicted output: \n" + str(np.matrix.round(NN.forward(X),2)))
    print("\n")
    NN.train(X,y)

NN.predict()