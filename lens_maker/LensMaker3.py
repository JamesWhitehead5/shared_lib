import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import multiprocessing as mp    

global CF # coupling field dataframe consisting 4 r data and phase


def phase_MSE(input_p_array, target_p_array):
    """calculates the difference of two phases. Consider the wrapping. """
    func = lambda d: abs(np.pi - abs((d-np.pi)%(2*np.pi))**2)
    total_difference = 0
    for (input_p, target_p) in zip (input_p_array, target_p_array):
        total_difference += func(input_p - target_p) 
    return total_difference


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
            print (' took %r  %2.2f s' % \
                  (method.__name__, (te - ts) ))
        return result
    return timed


def optimize_one(pixel):
    """a helper function for parallel computing"""
    return pixel.optimize()


class Pixel():
    """Each pixel has a target phase(s), actual phase(s), 
    transparency and a pillar width, and position"""

    def __init__(self, target_phases, position):
        """initialize the pixel"""
        self.target_phases = target_phases
        self.position = position
        self.r = None
        self.actual_phases = None
        self.sim = None
        self.wavelengths = None
       
    
    def optimize(self):
        """optimize (find the best r) for a single pixel"""

        
        # define error function
        err_func = lambda delta_p: (abs((delta_p-np.pi)%(2*np.pi) - np.pi))**2

        df = self.sim.copy()
        # for each wavelength
        for index, w in enumerate(self.wavelengths):    
            err = lambda x: err_func(eval(x)[0] - self.target_phases[index])
            # a new err function lol
            df['p_'+str(w)] = df['p_'+str(w)].apply(err)

        # calculate total error
        df['total_err'] = df['p_'+str(self.wavelengths[0])]
        for index, w in enumerate(self.wavelengths):
            if index > 0:
                df['total_err'] += df['p_'+str(w)]

        self.r = df.nsmallest(1, 'total_err')['r1'] 

        return self
    
    

class Lens():
    """A lens object containing pixels"""

   
    def __init__(self, **kwargs):
        """let's initiate this thing"""
        
        allowed_keys = [
            'lens_size', 'periodicity', 'min_tran', 'verbosity',
            'wavelengths', 'phase_mesh','lens_name', 'file_dir', 'metadata'
        ]
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
        self.CF = None       
        self.periodicity = float(self.periodicity)
    
 
    def load_coupling_field(self, df):
        """load coupling field dataframe"""
        self.CF = df
        pass

        
    def evaluate_coulping_field(self, r, target_phase):
        """evaluate the coupling field"""
        input_phase = self.CF.copy()
        input_phase = input_phase.query('r1=={}'.format(r[0])).query('r2=={}'.format(r[1])).query('r3=={}'.format(r[2])).query('r4=={}'.format(r[3]))
        input_phase['p_avg'] 

    @timeit
    def load_phase(self, phase_function):
        """load the phase function and initiate the pixels"""
        
        print 'Loading phase ...',
        
        #vectorize the phase_function for np
        phase_function = np.vectorize(phase_function)
        
        #calculate phase
        pixel_array_size = int(self.lens_size / self.periodicity)
        self.pixel_array_size = pixel_array_size
        actual_lens_size = pixel_array_size*self.periodicity
        phase_array_size = int(self.lens_size / self.periodicity)*self.phase_mesh
        half_lens_size = actual_lens_size/2.
        
        x = y = np.linspace(-half_lens_size, half_lens_size, pixel_array_size*self.phase_mesh)
        xx, yy = np.meshgrid(x, y, sparse=True) # position array
        
        #take the average of the subpixels
        phi_array = []
        m = self.phase_mesh
        for wavelength in self.wavelengths:
            target_phase_dense = phase_function(xx, yy, wavelength)
            target_phase_sparse = [[ [] for x in range(pixel_array_size)] for y in range(pixel_array_size)]
            for x in range(pixel_array_size):
                for y in range(pixel_array_size):
                    target_phase_sparse[x][y] = np.mean(target_phase_dense[m*x:m*(x+1), m*y:m*(y+1)])%(2*np.pi)
            phi_array.append(target_phase_sparse)
        
        if self.verbosity >= 2:
            plt.imshow(phi_array[0])
            plt.show()
            plt.imshow(phi_array[1])
            plt.show()
            plt.imshow(phi_array[2])
            plt.show()
                    
        #initiate pixels
        x = y = np.linspace(-half_lens_size, half_lens_size, pixel_array_size)
        xx, yy = np.meshgrid(x, y, sparse=True) # position array
        
        self.pixel_array = []
        for x in range(pixel_array_size):
            for y in range(pixel_array_size):
                # make pixels
                phi_s = []
                for index, _ in enumerate(self.wavelengths):
                    phi_s.append(phi_array[index][x][y])
                P = Pixel(phi_s, [xx[0][x], yy[y][0]])
                self.pixel_array.append(P)

        print '[DONE]',

    @timeit
    def load_simulation(self, df):
        """load RCWA simulations"""
        
        print 'Loading simulation ...',
        self.sim = df.copy()
        for w in self.wavelengths:
            self.sim = self.sim[self.sim['t_'+str(w)] >= self.min_tran]
    
        for P in self.pixel_array:
            P.sim = self.sim
            P.wavelengths = self.wavelengths
        print '[DONE]',
    
    

    @timeit
    def start(self):
        """Start to assemble"""
        
        print 'Starting assembly ...',
        
        p = mp.Pool(mp.cpu_count())
        ret_array  = p.map(optimize_one, self.pixel_array)
        p.close() # prevent memory leakage 
        p.join() # synchronization point
        
        self.pixel_array = ret_array
        
        print '[DONE]',
         
            
    def write(self, subdir = 'lens_name'):
        """
        Write lens the x.out, y.out, r.out
        Also creates images
        """

        Dir = self.file_dir
        Name = subdir

        directory = Dir + '/' + Name
        if not os.path.exists(directory):
            os.makedirs(directory)


        #setup data output files
        f_x = open(Dir+'/'+Name+"/x.out", "w+") 
        f_y = open(Dir+'/'+Name+"/y.out", "w+") 
        f_r = open(Dir+'/'+Name+"/r.out", "w+")


        lim = (self.lens_size + self.periodicity)/2.
        fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
        plt.axis([-lim,lim,-lim,lim])
        ax.set_aspect(1.)
        for P in self.pixel_array:
            x, y, r = P.position[0], P.position[1], P.r.values[0]
            circle = plt.Circle((x, y), r, color='r')
            ax.add_artist(circle)
            
            f_x.write(str(x)+ '\n')
            f_y.write(str(y)+ '\n')
            f_r.write(str(r)+ '\n')
        
        plt.savefig(directory+'/Lens.png')
        plt.show()

        #close data files
        f_x.close()
        f_y.close()
        f_r.close()
        
        # save metadata
        f_c = open(Dir+'/'+Name+"/lens_config.txt", "w+")
        attrs = [
                'lens_size', 'periodicity', 'min_tran', 'verbosity',
                'wavelengths', 'phase_mesh','lens_name', 'file_dir', 'metadata'
                ]
        for attr in attrs:
            f_c.write("obj.%s = %r" % (attr, getattr(self, attr)) + '\n')
        f_c.close()
            
