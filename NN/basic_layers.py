import theano
import theano.tensor as tt
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#from activations import share, init_wts, init_weights
import numpy as np

SEED = 05262015
rs = np.random.RandomState(SEED)

"""
Basic, fully connected layers
"""

##### UTILS
def init_wts(*argv):
    return 1 * (np.random.rand(*argv) - 0.5)

#From Xavier Bouthiller
def init_weights(n_in, n_out):
    rng = np.random.RandomState([2014, 03, 24])
    irange = np.sqrt(1. / (n_in))
    W = np.asarray(rng.uniform(-irange, irange, size=(n_in, n_out)),
                           dtype=theano.config.floatX)
    return W
   
def share(array, name=None, borrow=False):
    return theano.shared(value=np.asarray(array, dtype=theano.config.floatX), name=name)
#####

def log_softmax(x):
   m = tt.max(x, axis=1).dimshuffle((0, 'x'))
   x_diff = x - m
   return x_diff - tt.log(tt.sum(tt.exp(x_diff), axis=1).dimshuffle((0, 'x')))

non_linearities = {
    "linear": lambda state_below: state_below, 
    "sigmoid": lambda state_below: tt.nnet.sigmoid(state_below), 
    "softmax": lambda state_below: tt.nnet.softmax(state_below),
    "tanh": lambda state_below: tt.tanh(state_below),
    "rectifier": lambda state_below: tt.switch(state_below>=0, state_below, 0),
    "rectifier_below0": lambda state_below: tt.switch(state_below>=-1, state_below, 0),
    "leaky_rectifier": lambda state_below: tt.switch(state_below>=0, state_below, 0.1*state_below),
    "leaky_rectifier_below0": lambda state_below: tt.switch(state_below>=0, state_below, 0.1*state_below)
}

"""
Basic, Feedforward layer, fully connected
"""
class FFLayer():
    def __init__(self, non_linearity, inpt, n_dims, n_units, W=None, p_drop=0.0, train=0.0):
        self.B = share(np.zeros(n_units), 'b')
        if W is None:
            self.W = share(init_weights(n_dims, n_units), 'w', borrow=True)
            self.params = [self.W, self.B]
            self.l2params = [self.W]
        #Tied weights
        else:
            self.W = W
            self.params = [self.B]
            self.l2params = []
        self.non_linearty = non_linearities[non_linearity]
        rng = RandomStreams(rs.randint(999999))
        i_mask = train * rng.binomial(n=1, p=(1 - p_drop), size=(inpt.shape[0], n_dims), dtype=theano.config.floatX) \
                 + (1.0 - train) * (1 - p_drop)

        self.nout = n_units
        self.nin = n_dims
        self.output= self.non_linearty(tt.dot(i_mask*inpt, self.W) + self.B)
        
    
    def get_state(self):
        values = []
        for p in self.params:
            values.append(p.get_value())
        return values

    def set_state(self,state):
        for param,values in zip(self.params,state):
            param.set_value(values)

"""
Basic, Feedforward layer, fully connected used to 
handle whole sequences in a single pass with scan.

-Uses a mask of same size as the sequence (this could be change to a mask of size (t, mb_size)) 
in order to ouput last valid out in zero-padded regions.
"""
class FFLayer_seq_mb():
    def __init__(self, 
                    non_linearity, 
                    inpt,
                    n_dims, 
                    seq_mask,
                    n_units, 
                    W=None):
        
        b = share(np.zeros(n_units), 'b')
        if W == None:
            w = share(init_weights(n_dims, n_units), 'w', borrow=True)
            self.params = [w, b]
            self.l2params = [w]
        #Tied weights
        else:
            w = W
            self.params = [b]
            self.l2params = []
            
        self.non_linearty = non_linearities[non_linearity]
        
        def step(in_t, mask_t, out_tm1):
            out = self.non_linearty(tt.dot(in_t, w) + b) # (mb, n_units)
            out = tt.shape_padright(mask_t[:,0],1) * out + tt.shape_padright((1. - mask_t)[:,0], 1) * out_tm1
            return out #(mb,ndims)
        
        #output for each t, (t, mb_size, n_units)
        rval, _ = theano.scan(
            step,
            sequences=[inpt, seq_mask],
            outputs_info=[tt.unbroadcast(tt.alloc(b, inpt.shape[1], n_units),1)]
        )

        self.output = rval #(t, mb_size, n_units)
        self.nout = n_units
        self.nin = n_dims
        self.W = w
        self.B = b
    
    def get_state(self):
        values = []
        for p in self.params:
            values.append(p.get_value())
        return values

    def set_state(self,state):
        for param,values in zip(self.params,state):
            param.set_value(values)            
            
"""
Basic, Softmax layer. Might not be useful anymore.
"""
class SoftmaxLayer():
    def __init__(self, inpt, in_sz, n_classes):
        b = share(np.zeros(n_classes), 'b_softmax')
        w = share(init_weights(in_sz, n_classes), 'w_softmax')
        self.output = tt.nnet.softmax(tt.dot(inpt, w) + b)
        self.outsum = self.output.sum(axis=1)
        self.params = [w, b]
        self.l2params = [w]
    
    def get_state(self):
        values = []
        for p in self.params:
            values.append(p.get_value())
        return values

    def set_state(self,state):
        for param,values in zip(self.params,state):
            param.set_value(values)


"""
Basic, Softmax layer that handles
whole sequences in a single pass with scan.
"""            
class SoftmaxLayer_mb():
    def __init__(self, 
                    inpt, #(n_steps, mb, n_dims)
                    in_sz, 
                    seq_mask,
                    n_classes):
    
        b = share(np.zeros(n_classes), 'b_softmax')
        w = share(init_weights(in_sz, n_classes), 'w_softmax')
        
        
        def step(in_t, mask_t, out_tm1):
            out = tt.nnet.softmax(tt.dot(in_t, w) + b) # (mb, n_classes)
            out = tt.shape_padright(mask_t[:, 0]) * out
            return out

        #output for each t, (t, mb_size, n_classes)
        rval, _ = theano.scan(
            step,
            sequences=[inpt, seq_mask],
            outputs_info=[tt.alloc(b, inpt.shape[1], n_classes)]
        )
        #Dims = (t, mb_size, n_classes)
        self.output = rval
        self.params = [w, b]
        self.l2params = [w]
    
    def get_state(self):
        values = []
        for p in self.params:
            values.append(p.get_value())
        return values

    def set_state(self,state):
        for param,values in zip(self.params,state):
            param.set_value(values)                          
            
"""
Softmax applied to the sum of the activations of a fully connected linear layer over time.
Used in Du et al. 2015 - Hierarchical Recurrent Neural Network for Skeleton Based Action Recognition
"""            
class SoftmaxOfSumLayer_mb():
    def __init__(self, 
                    inpt, #(n_steps, mb, n_dims)
                    in_sz, 
                    seq_mask,
                    n_classes):
    
        b = share(np.zeros(n_classes), 'b_softmax')
        w = share(init_weights(in_sz, n_classes), 'w_softmax')
        
        def step(in_t, mask_t, out_tm1):
            out = (tt.dot(in_t, w) + b) # (mb, n_classes)
            #0 everywhere if seq_mask = 0.
            out = mask_t[:,0].dimshuffle(0,'x') * out
            return out

        #output for each t, (t, mb_size, n_classes)
        rval, _ = theano.scan(
            step,
            sequences=[inpt, seq_mask],
            outputs_info=[tt.alloc(b, inpt.shape[1], n_classes)]
        )
        A = tt.sum(rval, axis=0) #sum on all time steps. Dims = (mb_size, n_classes)
        #self.output = tt.log(tt.nnet.softmax(A))
        self.output = log_softmax(A)
        self.params = [w, b]
        self.l2params = [w]
    
    def get_state(self):
        values = []
        for p in self.params:
            values.append(p.get_value())
        return values

    def set_state(self,state):
        for param,values in zip(self.params,state):
            param.set_value(values)

"""
Same as above, but for binary classification (e.g. a GAN discriminator)
"""
class SigmoidOfSumLayer_mb():
    def __init__(self,
                 inpt,  # (n_steps, mb, n_dims)
                 in_sz,
                 seq_mask):

        b = share(np.zeros(1), 'b_sigmoid')

        #One output unit:
        w = share(init_weights(in_sz, 1), 'w_sigmoid')

        def step(in_t, mask_t, out_tm1):
            out = (tt.dot(in_t, w) + b)  # (mb, 1)
            # 0 everywhere if seq_mask = 0.
            # out = mask_t[:, 0].dimshuffle(0, 'x') * out
            return out

        # output for each t, (t, mb_size, 1)
        rval, _ = theano.scan(
            step,
            sequences=[inpt, seq_mask],
            outputs_info=[tt.alloc(b, inpt.shape[1], 1)]
        )
        # Sum on all time steps. Dims = (mb_size, 1)
        # Should it be the mean?
        A = tt.sum(rval, axis=0)

        self.output = tt.nnet.sigmoid(A)
        self.params = [w, b]
        self.l2params = [w]

    def get_state(self):
        values = []
        for p in self.params:
            values.append(p.get_value())
        return values

    def set_state(self, state):
        for param, values in zip(self.params, state):
            param.set_value(values)


class ConcatLayer_mb():
    def __init__(self,
                seq1, #(n_steps, mb, n_dims)
                seq2,  #(n_steps, mb, n_dims)
                nTotal,
                ):
        #Concat over dims:
        self.output = tt.concatenate([seq1, seq2], axis=2)
        self.nout = nTotal
        self.l2params = []
        self.params = []

    def get_state(self):
        values = []
        return values

    def set_state(self, state):
        pass


