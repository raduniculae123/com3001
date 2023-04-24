#!/usr/bin/env python
# coding: utf-8

# In[10]:


#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import numpy.matlib 
import math
import matplotlib.pyplot as plt
import csv
from matplotlib import cm


# In[46]:


class Numerical_methods:
    
    def __init__(self,f ):  
        
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


# In[47]:



def f(x,t):    
    
    b = 0.5 #infection rate
    k = 0.33 #recovery rate
    
    
    z=np.zeros([np.shape(x)[0]])
    
    z[0] = -b * x[0] * x[1]
    z[1] = b * x[0] * x[1] - k*x[1]
    z[2] = k * x[1]
    
    return z


# In[51]:


NM=Numerical_methods(f)           ## Object definition


dt = 0.2                      ## Value of dt
N_iter = 1000               ## Number of iteration. 


X_RK2=np.zeros([2,N_iter])


   
t_start=0

#initial conditions of the document - 7.9 mil Susceptible and 10 Infected
NM.Initialise([1, 1.27 * 10**-6, 0], t_start)     ## Setting the initial conditions in the object
X_RK2,ts=NM.RungeKutta2(dt,N_iter)

fig1 = plt.figure()  # Create a new figure for the first 3D plot
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot3D(X_RK2[0,:], X_RK2[1,:], X_RK2[2,:], 'red')

fig2, ax2 = plt.subplots() # one axis on figure
ax2.plot(ts, X_RK2[0,:], color='blue', label='Susceptible (S)')
ax2.plot(ts, X_RK2[1,:], color='orange', label='Infected (I)')
ax2.plot(ts, X_RK2[2,:], color='green', label='Recovered (R)')
ax2.legend()  # Display the legend on the plot


#new initial conditions (2nd scenario) - 7.9 mil Susceptible and 10000 Infected
NM.Initialise([1, 1.27 * 10**-6 * 1000, 0], t_start)     
X_RK2,ts=NM.RungeKutta2(dt,N_iter)

fig3 = plt.figure() 
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot3D(X_RK2[0,:], X_RK2[1,:], X_RK2[2,:], 'red')


fig4, ax4 = plt.subplots() # one axis on figure
ax4.plot(ts, X_RK2[0,:], color='blue', label='Susceptible (S)')
ax4.plot(ts, X_RK2[1,:], color='orange', label='Infected (I)')
ax4.plot(ts, X_RK2[2,:], color='green', label='Recovered (R)')
ax4.legend()  



#new initial conditions (3rd scenario) - 7.9 mil Susceptible and 1 Infected
NM.Initialise([1, 1.27 * 10**-6 /10 , 0], t_start)     
X_RK2,ts=NM.RungeKutta2(dt,N_iter)

fig5 = plt.figure() 
ax5 = fig5.add_subplot(111, projection='3d')
ax5.plot3D(X_RK2[0,:], X_RK2[1,:], X_RK2[2,:], 'red')


fig6, ax6 = plt.subplots() # one axis on figure
ax6.plot(ts, X_RK2[0,:], color='blue', label='Susceptible (S)')
ax6.plot(ts, X_RK2[1,:], color='orange', label='Infected (I)')
ax6.plot(ts, X_RK2[2,:], color='green', label='Recovered (R)')

plt.show()



