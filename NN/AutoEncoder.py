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
    def __init__(self, n_dims, n_units, ae_nl, learning_rate=0.001, L2 =0.01, p_drop=0.0):
        
        #input sequence
        inputs = tt.matrix('input') #matrix of possibly noisy minibatch n_mb x n_dims
        targets = tt.matrix('target') #vector of clean outputs
      
        self.layers = []
        self.encoder_layers = []

        ##ENCODER
        lay1 = FFLayer(
            ae_nl,
            inputs,
            n_dims,
            n_units[0])  
        self.layers.append(lay1)
        self.encoder_layers.append(lay1)
        
        for i in range(1, len(n_units)):
            lay = FFLayer(
                ae_nl,
                self.layers[-1].output,
                self.layers[-1].nout,
                n_units[i])
            self.layers.append(lay)
            self.encoder_layers.append(lay)
        
        self.representation = self.layers[-1].output
        
        ##DECODER
        #take FFlayers in reverse order, with tied weights
        for layer in self.layers[::-1]:
            new_lay = FFLayer(
                ae_nl,
                self.layers[-1].output,
                self.layers[-1].nout,
                layer.nin,
                W=layer.W.T)
            self.layers.append(new_lay)
        
        self.decoder = self.layers[-1]

        # L2 regularization
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

    def get_encoder_states(self):
        states = []
        for lyr in (self.encoder_layers):
            states.append(lyr.get_state())
        return states

    def set_states(self, states):
        for (lyr,state) in zip(self.layers, states):
            lyr.set_state(state)


class AutoEncoderRegressor():
    """
    Simple AE for unsupervised learning.
    params:
        n_dims: number of dimensions of inputs
        n_units: array of hidden units. Tells how many layers
        n_classes: number of classes
    """

    def __init__(self, n_dims, encode_units, regress_units, ae_nl, r_nl, learning_rate=0.001, L2 =0.01, p_drop=0.0):

        # input sequence
        inputs = tt.matrix('input')  # matrix of possibly noisy minibatch n_mb x n_dims
        clean_inputs = tt.matrix('target')  # vector of clean outputs
        targets = tt.matrix('target')  # vector of values to regress
        recon_ratio = tt.scalar('recon_ratio')  # reconstruction ratio (do we focus the loss on reconstruction or regression?)
        train_indicator = tt.scalar('train indicator')

        self.layers = []
        self.encoder_layers = []

        # ENCODER :
        lay1 = FFLayer(
            ae_nl,
            inputs,
            n_dims,
            encode_units[0])
        self.layers.append(lay1)
        self.encoder_layers.append(lay1)

        for i in range(1, len(encode_units)):
            lay = FFLayer(
                ae_nl,
                self.layers[-1].output,
                self.layers[-1].nout,
                encode_units[i])
            self.layers.append(lay)
            self.encoder_layers.append(lay)

        self.representation = self.layers[-1].output

        # DECODER (RECONSTRUCTION)
        # take FFlayers in reverse order, with tied weights
        for layer in self.layers[::-1]:
            new_lay = FFLayer(
                ae_nl,
                self.layers[-1].output,
                self.layers[-1].nout,
                layer.nin,
                W=layer.W.T)
            self.layers.append(new_lay)
        self.decoder = self.layers[-1]

        # REGRESSOR :
        lay = FFLayer(
            r_nl,
            self.encoder_layers[-1].output,
            self.encoder_layers[-1].nout,
            regress_units[0],
            p_drop=p_drop,
            train=train_indicator)
        self.layers.append(lay)

        for i in range(1, len(regress_units)):
            lay = FFLayer(
                r_nl,
                self.layers[-1].output,
                self.layers[-1].nout,
                regress_units[i],
                p_drop=p_drop,
                train=train_indicator)
            self.layers.append(lay)

        #Last layer, one unit (predicted scalar):
        self.regressor = FFLayer(
            'linear',
            self.layers[-1].output,
            self.layers[-1].nout,
            1,
            p_drop=p_drop,
            train=train_indicator)
        self.layers.append(self.regressor)

        # L2 regularization
        sum_p = 0.0
        for lyr in (self.layers):
            for param in lyr.l2params:
                sum_p += tt.sum(param.get_value() ** 2)

        self.l2_cost = L2*sum_p
        self.AE_cost = tt.sum(tt.mean((self.decoder.output - clean_inputs) ** 2, axis=0))
        self.regress_cost = tt.mean((self.regressor.output - targets) ** 2)
        self.cost = self.l2_cost + self.AE_cost*recon_ratio + self.regress_cost*(1.0-recon_ratio)

        # Updates
        params = []
        grads = []
        for lyr in (self.layers):
            for param in lyr.params:
                params.append(param)
                grads.append(tt.grad(self.cost, param))
        updates = optim.Adam(params, grads, lr=learning_rate)

        self.train = theano.function(
            inputs=[inputs, clean_inputs, targets, recon_ratio, train_indicator],
            outputs=[
                self.regressor.output,
                self.decoder.output,
                self.AE_cost,
                self.regress_cost,
                self.cost],
            updates=updates,
            allow_input_downcast=True,
        )

        self.test = theano.function(
            inputs=[inputs, clean_inputs, targets, recon_ratio, train_indicator],
            outputs=[
                self.regressor.output,
                self.decoder.output,
                self.AE_cost,
                self.regress_cost,
                self.cost],
            allow_input_downcast=True)

        self.predict = theano.function(
            inputs=[inputs, train_indicator],
            outputs=[
                self.regressor.output,
                self.decoder.output],
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
        for (lyr, state) in zip(self.layers, states):
            lyr.set_state(state)

    def set_encoder_states(self, states):
        for (lyr, state) in zip(self.encoder_layers, states):
            lyr.set_state(state)