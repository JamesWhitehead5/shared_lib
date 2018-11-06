import pandas as pd
import numpy as np
from ast import literal_eval


def to_r(CSCS):
    """CSCS is a string, returns [r1, r2, r3, r4]"""

    f_list = CSCS.split('C')[1:]
    r = []
    for ele in f_list:
        r.append(eval(ele[:ele.find(')')+1])[2])
    return r


def extract_features(df_test, subdivision = 2, n_r = 4):

    # convert the CSCS to useful features
    for index, row in df_test.iterrows():
        r = to_r(row['CSCS'])
        df_test.at[index,'r1'] = r[0]
        
        if n_r == 4:
            df_test.at[index,'r2'] = r[1]
            df_test.at[index,'r3'] = r[2]
            df_test.at[index,'r4'] = r[3]
            #df_test.at[index,'r5'] = r[4]
        
#         if type(row['transmission']) == float:
#             t = row['transmission']
#             df_test.at[index,'p_avg'] = row['phase']
#             df_test.at[index,'t_avg'] = row['transmission']

        else:
            e = np.array(eval(row['e_field']))

            a = np.arange(0,subdivision)
            ex_mean = np.array([])
            sub_size = np.shape(e)[0]/subdivision
            for sub_y in a:
                for sub_x in a:
                    sub_efield = e[sub_x*sub_size:(sub_x+1)*sub_size, 
                                        sub_y*sub_size:(sub_y+1)*sub_size]
                    ex_mean = np.append(ex_mean, np.mean(sub_efield))
            t = np.abs(ex_mean)
            p = np.arctan2(ex_mean.imag, ex_mean.real) 
            p = np.mod(p, 2*np.pi)
            df_test.at[index,'p_avg'] = str(p.tolist())
            df_test.at[index,'t_avg'] = str(t.tolist())
        
    return df_test


def to_lambda(CSCS):
    s = CSCS.split('-')
    return eval(s[1])

def uppack_CSCS(df_test):
    for index, row in df_test.iterrows():
        w = to_lambda(row['CSCS'])
        df_test.at[index,'wavelength'] = w
    return df_test