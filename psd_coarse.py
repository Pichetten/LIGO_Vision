# -*- coding: utf-8 -*-
"""PSD_course.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Jsg38i4VacamxQXAsc4cKgIqn_ld7j7X
"""

import numpy as np
import matplotlib.pyplot as plt

def PSD_coarse(PSD,f_course):

    """ This function inputs the full resolution PSD and
        the coursened frequency array. The output is a 
        course PSD evaluated at all points in the course
        frequency array."""

    N_full = len(PSD)
    N_course = len(f_course)
    filler_vec = np.zeros(N_course)
    PSD_f = PSD[:,1].tolist()
    PSD_course = np.zeros(N_course)
    both = set(PSD_f).intersection(f_course)
    indx_PSD = sorted([PSD_f.index(x) for x in both])
    indx_f_course = sorted([f_course.index(x) for x in both])
    
    for i in range(N_course):
        PSD_course[i] = PSD[indx_PSD[i],0]  
       
    

    return PSD_course

f_full = list(range(1000))
PSD_full = np.vstack((np.linspace(1e-38,1e-40,1000),f_full)).T
f_course = f_full[::50]

PSD_new = PSD_course(PSD_full,f_course)

plt.plot(f_full,PSD_full[:,0], 'g^', label = "original PSD")
plt.plot(f_course,PSD_new, 'r^', label = "course PSD")
plt.legend()
plt.grid('True')
