import numpy as np
import numpy.random as npr

# This is the Python class that implements the main XOR neural network that must be trained
class DFA_NN:
    
    def __init__(self,sigma=1):
        self.inputs = np.zeros(2)
        self.z_hidden = np.zeros(2)
        self.a_hidden = np.zeros(2)
        self.z_output = np.zeros(1)
        self.output = np.zeros(1)
        
        self.W_hidden = np.random.normal(0,sigma,size=(2,2))
        self.b_hidden = np.random.normal(0,sigma,size=2)
        self.W_output = np.random.normal(0,sigma,size=2)
        self.b_output = np.random.normal(0,sigma,size=1)
        
    def sigmoid(self,x):
        return 1./(1. + np.exp(-x))
        
    def update(self):
        self.z_hidden = np.dot(self.W_hidden,self.inputs)+self.b_hidden
        self.a_hidden = self.sigmoid(self.z_hidden)
        self.z_output = np.dot(self.W_output,self.a_hidden)+self.b_output
        self.output = self.sigmoid(self.z_output)
    
    def forward_pass(self,inputs):
        self.inputs = inputs
        self.update()

