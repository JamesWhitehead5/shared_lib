# -*- coding: utf-8 -*-
#!/usr/bin/env python

__author__ = "Luocheng Huang"
__copyright__ = "Copyright 2018"
__credits__ = ["Luocheng Huang"]
__license__ = "GPL"
__version__ = "2.0.2"
__maintainer__ = "Luocheng Huang"
__email__ = "luocheng@uw.edu"
__status__ = "Production"

import S4
import pandas as pd
import os
import numpy as np
import math
import time as time
from multiprocessing import Pool
import matplotlib.pyplot as plt

CWD = os.getcwd()+'/'


    
def simulate_one(config):
    """a wrapper for pickling
    simulates one instance
    """
    new = RCWA.Simulation()
    new.load_input(config)

    return new.run()


class RCWA:
    """
    An rcwa simulation ojbect
    """
    
    def __init__(self, input_list=None, cores=10, field = None):
        self.input = input_list
        self.cores = cores
        self.field = field
        global FIELD
        """field is the amount of field in the center measured. 1 is 100% .5 is 25%
            returns the mean value if field==None
            else returns the center field."""
        FIELD = field
        pass
    
    
    def timeit(method):
        """a decor for timing methods, should be used as a decorator
        """
        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            if 'log_time' in kw:
                name = kw.get('log_name', method.__name__.upper())
                kw['log_time'][name] = int((te - ts) * 1000)
            else:
                print '%r  %2.2f s' % \
                      (method.__name__, (te - ts) )
            return result
        return timed


    @timeit
    def simulate_and_graph(self):
        """
        inputs the attribute list, 
        :return: simulate results
        """

        p = Pool(self.cores)
        df = pd.DataFrame()
        FIELD = self.field
        solution = p.map(simulate_one, self.input)  # create a pool of workers and map it onto different cores
        p.close() # prevent memory leakage 
        p.join() # synchronization point

        t_list = []
        p_list = []
        for s in solution:
            t_list.append(s[0])
            p_list.append(s[1])
        # make phase continuous
        # new_p = unwrap(p_list)
        new_p = [x[0] / (2*math.pi) for x in p_list]

        config = self.input[0]

        # plot data
        plt.figure(0)
        plt.figure(figsize=(10, 5), dpi=200)
        label1 = 'phase'
        label2 = 'amplitude'
        plt.plot(new_p, label=label1)
        plt.plot(t_list, label=label2)
        plt.xlabel('Grating post width')
        plt.ylabel('amp & phase')
        plt.title('Wavelength = {} nm'.format('unknown'))
        plt.legend()


        plt.savefig("{}.png".format(('unknown')))
        plt.show();
        plt.clf();

        # save to csv
        a = (new_p, t_list)

        np.savetxt("data_{}.csv".format(('unknown')), a, delimiter=",")


        pass

    
    @timeit
    def simulate(self):
        """
        inputs the attribute list, 
        :return: simulate results
        """

        p = Pool(self.cores)
        df = pd.DataFrame(self.input)
        solution = p.map(simulate_one, self.input)  # create a pool of workers and map it onto different cores
        solutions = pd.DataFrame(solution)
        p.close() # prevent memory leakage 
        p.join() # synchronization point

        result = pd.concat([df, solutions], axis=1)
        result.columns = ['CSCS', 'e_field']

        # make phase continuous
        # new_p = unwrap(p_list)

        return result
        

    
    

    class Simulation:
        """A simulation object"""
        
        def __init__(self):
            self.layers = []
            self.basis = ((1,0),(0,1))
            self.NumBasis=150
            self.layers = None
            self.layer_material = None
            self.layer_thickness = None
            self.layer_pattern = None
            self.transmission = None
            self.wavelength = None
            self.buffer = 0.1
            self.Nxy = 52
        
        
        def load_input(self,input_instance):
            """loades the canonical id into local variables"""
            
            
            # the first step is to assign values to the variables from the input_instance
            parts = input_instance.split('/')
            first_arg = parts[0].split('-')
            if first_arg[0] == 't':
                self.transmission = True
            elif first_arg[0] == 'r':
                self.transmission = False
            
            self.wavelength = float(first_arg[1])
            self.basis = eval(first_arg[2])
            
            self.layers = parts[1:]
            layers_split = [layer.split('=') for layer in self.layers]
            self.layer_material = [layer_split[0] for layer_split in layers_split]
            split_2 = [layer_split[1] for layer_split in layers_split]
            self.layer_thickness = []
            self.layer_pattern = []
            for x in split_2:
                x_split = x.split(':')
                if len(x_split) == 1:
                    self.layer_thickness.append(float(x_split[0]))
                    self.layer_pattern.append(None)
                else:
                    self.layer_thickness.append(float(x_split[0]))
                    self.layer_pattern.append(x_split[1].split('-'))
                    
                    

        
        def run(self):
            """create layers and return a S4 object
            :type attr: dict
            return S
            """
            def querry_n(material, wavelength):
                """
                given the material in string, wavelength in µm
                output (n,k)
                """
                mat = pd.read_csv(CWD+'n_data/{}_n.csv'.format(material))

                w = mat['Wavelength, µm']
                n_list = mat['n']
                k_list = mat['k']

                n = np.interp(wavelength, w, n_list) # interpolate the refractive index value
                k = np.interp(wavelength, w, k_list) # interpolate the k value

                return n + k*1j      

            # define the S object
            S = S4.New(Lattice=self.basis,
                       NumBasis=self.NumBasis)  # orthogonal basis, numbasis scales with memory O(N^2)

            # define materials
            for material in set(self.layer_material):
                n = querry_n(material,self.wavelength)
                # defines permittivity in terms of refractive index
                eps = n**2
                S.SetMaterial(Name=material, Epsilon=eps)
            S.SetMaterial(Name='Vacuum', Epsilon=1.0)

            # define layers
            S.AddLayer(Name='air_above', Thickness=self.buffer, Material='Vacuum')
            index = 0
            for layer in zip(self.layer_material, self.layer_thickness, self.layer_pattern):
                index += 1
                if layer[2] == None:
                    S.AddLayer(Name=str(index), Thickness=layer[1], Material=layer[0])
                else:
                    S.AddLayer(Name=str(index), Thickness=layer[1], Material='Vacuum')
                    if '+' in layer[2]:
                        patterns = layer[2].split('+')
                    else:
                        patterns = layer[2]
                    for pattern in patterns:
                        shape = pattern[0]
                        coor = eval(pattern[1:])
                        if shape == 'C':
                            # create circle features
                            if coor[2] != 0:
                                S.SetRegionCircle(
                                    Layer = str(index),
                                    Material = layer[0],
                                    Center = (coor[0],coor[1]),
                                    Radius = coor[2]
                                )
                        elif shape == 'S':
                        # create circle features
                            if coor[2] != 0:
                                S.SetRegionSquare(
                                    Layer = str(index),
                                    Material = layer[0],
                                    Center = (coor[0],coor[1]),
                                    Angle = 0,
                                    Halfwidths = (coor[2], coor[2])
                                    
                                )
            
            S.AddLayer(Name='below', Thickness=self.buffer, Material='Vacuum')


            # simulation settings
            S.SetExcitationPlanewave(IncidenceAngles=(0.0, 0.0), sAmplitude=0.0,
                                     pAmplitude=1.0, Order=0)

            S.SetFrequency(1/self.wavelength)

            # For higher accuracy
            S.SetOptions(SubpixelSmoothing=True)
            
            # e is array of dimension Nx by Ny, each element is a tuple of length 3
            
            if  not self.transmission:
                z_pos = self.buffer/(2.0)
            else:
                z_pos = self.buffer*1.5+sum(self.layer_thickness)
            
            e, h = S.GetFieldsOnGrid(z=z_pos, NumSamples=(self.Nxy, self.Nxy),
                                     Format='Array')  
            e = np.array(e)
            if not self.transmission:
                e = e - np.exp(1j*2*np.pi/self.wavelength*self.buffer/2)
                
            if FIELD==None:
                e_field = np.mean(e[:,:,0])  # adding 1 removes incident wave 
            else:                
                (a,b,_) = np.shape(e)
                edge = (1-FIELD)/2.0
                e_field = e[int(a*edge):int(a*(1-edge)),int(b*edge):int(b*(1-edge)),0]
                
            e_field = np.round(e_field, 3)
            
            return pd.Series(str(e_field.tolist()))