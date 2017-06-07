from __future__ import print_function

import sys
import theano
import theano.tensor as tt
import theano.gradient as tg
import numpy as np
from basic_layers import FFLayer
from activations import share
import Optimizers as optim

class AutoEncoder():
    """
    Simple AE for unsupervised learning. 
    params:
        n_dims: number of dimensions of inputs
        n_units: array of hidden units. Tells how many layers
        n_classes: number of classes
    """
    def __init__(self, n_dims, n_units, nl, learning_rate=0.001):
        
        #input sequence
        inputs = tt.matrix('input') #matrix of possibly noisy minibatch n_mb x n_dims
        targets = tt.matrix('target') #vector of clean outputs
      
        self.layers = []

        ##ENCODER
        lay1 = FFLayer(
            nl,
            inputs,
            n_dims,
            n_units[0])  
        self.layers.append(lay1)
        
        for i in range(1, len(n_units)):
            lay = FFLayer(
                nl,
                self.layers[-1].output,
                self.layers[-1].nout,
                n_units[i])
            self.layers.append(lay)
        
        self.representation = self.layers[-1].output
        
        ##DECODER
        #take FFlayers in reverse order, with tied weights
        for layer in self.layers[::-1]:
            new_lay = FFLayer(
                nl,
                self.layers[-1].output,
                self.layers[-1].nout,
                layer.nin,
                W=layer.W.T)
            self.layers.append(new_lay)
        
        self.decoder = self.layers[-1]

        # L2 regularization
        L2=0.1
        sum_p = 0.0
        for lyr in (self.layers):
            for param in lyr.l2params:
                sum_p += tt.sum(param.get_value() ** 2)

        self.cost = tt.sum(tt.mean((self.decoder.output - targets)**2, axis=0)) + L2*sum_p
            
        #Updates
        params = []
        grads = []
        for lyr in (self.layers):
            for param in lyr.params:
                params.append(param)
                grads.append(tt.grad(self.cost, param))
        updates = optim.Adam(params, grads, lr=learning_rate)
        
        self.train = theano.function(
            inputs=[inputs, targets],
            outputs=[
                self.representation,
                self.decoder.output,
                self.cost],
            updates=updates,
            allow_input_downcast=True,
            )

        self.test = theano.function(
            inputs=[inputs, targets],
            outputs=[
                self.representation,
                self.decoder.output,
                self.cost],
            allow_input_downcast=True)
            
        self.rep = theano.function(
            inputs=[inputs],
            outputs=[self.representation],
            allow_input_downcast=True)

    def get_states(self):
        states = []
        for lyr in (self.layers):
            states.append(lyr.get_state())
        return states

    def set_states(self, states):
        for (lyr,state) in zip(self.layers, states):
            lyr.set_state(state)
