import pickle
from flask import Flask, request
from flasgger import Swagger
import numpy as np
import pandas as pd

#reading the binaryversion of python script
with open('E:/ATISHAY/C/major 1/random forest basic app/rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

    
    
    
    
    