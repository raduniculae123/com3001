#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numpy.matlib 
import math
import matplotlib.pyplot as plt
import csv
from matplotlib import cm


class Numerical_methods:
    
    def __init__(self,f):  
        
        self.f=f
        self.x=[]
        self.t=[]
        self.N_dim=[]
        
        
    def Initialise(self,x_start,t_start):
        
        
        self.N_dim=np.shape(x_start)[0]
        self.x=x_start
        self.t=t_start
        
        
    
    def RungeKutta2(self,dt,N_iter):
        
        X=np.zeros([self.N_dim,N_iter])
        T=np.zeros([N_iter])
        
        X[:,0]=np.copy(self.x)
        T[0]=np.copy(self.t)
        
        for n in range(1,N_iter):
            
            k1=f(self.x,self.t)
            k2=f(self.x+dt*k1,self.t+dt)    
                        
            self.x=self.x+dt*(k1+k2)/2
            
            X[:,n]=np.copy(self.x)
            
            self.t=self.t+dt
            T[n]=np.copy(self.t)
            
        return X, T
       

                
def f(x,t):     ## Definition of the function for SIR. t needs to be an input even if the function does not use it, as in this case.
                ## This formulation permits us to be general and to apply the same code to non autonomous systems.
    
    b = 1/2
    k = 1/3
    
    
    z=np.zeros([np.shape(x)[0]])
    
    z[0] = -b * x[0] * x[1]
    z[1] = b * x[0] * x[1] - k*x[1]
    z[2] = k * x[1]
    
    return z

## As in the previous lab, use the class and the function above to complete the exercise...

NM=Numerical_methods(f)           ## Object definition


dt = 0.2                      ## Value of dt
N_iter = 1000               ## Number of iteration. 


X_RK2=np.zeros([2,N_iter])


x_start=[1, 1.27 * 10**-6, 0]     
t_start=0.


NM.Initialise(x_start,t_start)     ## Setting the initial conditions in the object
X_RK2,ts=NM.RungeKutta2(dt,N_iter) 



ax = plt.axes(projection='3d')

ax.plot3D(X_RK2[0,:], X_RK2[1,:], X_RK2[2,:], 'red')



# %%
