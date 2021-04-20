import numpy as np
import numpy.random as npr

# This is the Python class that implements the error approximating network with pre-trained weights and biases imported from another file
class err_approx:
    
    def __init__(self,model):
        self.W1 = model.layers[0].get_weights()[0]
        self.b1 = model.layers[0].get_weights()[1]
        self.W2 = model.layers[1].get_weights()[0]
        self.b2 = model.layers[1].get_weights()[1]
        self.W3 = model.layers[2].get_weights()[0]
        self.b3 = model.layers[2].get_weights()[1]
            
    def sigmoid(self,x):
        return 1./(1. + np.exp(-x))
    
    def forward_pass(self,inputs):
        hidden1 = self.sigmoid(np.dot(self.W1.T,inputs)+self.b1)
        hidden2 = self.sigmoid(np.dot(self.W2.T,hidden1)+self.b2)
        return (np.dot(self.W3.T,hidden2)+self.b3)[0]