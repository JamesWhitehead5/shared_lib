import S4
import os
import numpy as np
#cwd = os.getcwd()
import sys
sys.path.append("../rcwa")
from RCWA import * 
from CSCS_to_features import *
#os.chdir(cwd)

Dir = 'test'
Name = '1'


# configuration
wavelengths = [0.633, 0.54, 0.46] # has to be in an array
periodicity = 0.7*0.633 # optimized for 560 nm
grat_tkn = 1
buff_tkn = 0.633
dbr_Si3N4 = 0.633 / 4.0 / 2.
dbr_SiO2  = 0.633 / 4.0 / 1.457

aspect_ratio = 6
resolution = 100


# don't worry about
max_r = periodicity/2.
min_r = grat_tkn/float(aspect_ratio)/2.
r_list = np.linspace(min_r, max_r, num=resolution)


# make a directory
directory = Dir + '/'
if not os.path.exists(directory):
    os.makedirs(directory)

subdivision = 1

# start RCWA
inputs = []
for w in wavelengths:
    for r in r_list:
        inpt = 'r-{}-({},0.),(0.,{})/TiO2={}:C(0,0,{})/SiO2={}/Si3N4={}'\
          .format(w,periodicity,periodicity,grat_tkn,r,dbr_SiO2,dbr_Si3N4)
        inputs.append(inpt)


#simulate_one(inputs[0], field=1)

new = RCWA(inputs, 19, field=1)
df = new.simulate()
