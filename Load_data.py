# -*- coding: utf-8 -*-
import numpy as np
import 

def load_data(valid_index=0, n_parts=10):
    # Load the data
    with open('Data.pkl', 'rb') as f:
                data = pickle.load(f)
                X = data['X']
                Y = data['Y']
    
    # Match example and label together and shuffle the whole dataset
    dataset = zip(X,Y)
    random.seed(1234)
    random.shuffle(dataset)
    
    return dataset
