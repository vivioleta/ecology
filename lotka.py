import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from scipy import integrate
import random


#Lotka-Volterra equations for a pair of predator-prey species
alpha = 1.
beta = 1.
delta = 1.
gamma = 1.
x0 = 4.
y0 = 2.
def lotka_prey(X, t, alpha, beta, delta, gamma):
    x, y = X
    dotx = x * (alpha - beta * y)
    doty = y * (-delta + gamma * x)
    return np.array([dotx, doty])

#Lotka-Volterra equations for n competing species
def lotka_competitive(P, t, r, k, alpha):
    dotP = r*P - (r/k)*P*(alpha@P)
    return dotP

#Lotka-Volterra equations for n competing species with Allee effects
def lotka_allee(P, t, r, k, a, alpha):
    np.fill_diagonal(alpha,P - a) #using this term for the diagonals to simplify the expression for the Lotka-Volterra equations
    dotP = r*P*(P - a) - (r/k)*P*(alpha@P)
    return dotP


#Solving function
def solve(r,k,alpha,a,P0,t):
    res = integrate.odeint(lotka_allee, P0, t, args = (r,k,a,alpha))
    return res


#Parameters Choosing
n = 4 #number of species
r = np.array([1,2,3,4]) #vector with unconstrained growh rates for each species
k = np.ones(n) #vector with the carrying campacities for each species
alpha = np.array([[1,0.4,0.7,0.8],[0.1,1,2,0.04],[0.3,0.5,1,0.9],[0.1,0.1,0.1,1]]) #matrix with the parameters for each possible species interaction, with alpha_ii = 1 for all i
a = np.ones(n)
P0 = np.ones(n) #initial values
Nt = 1000
tmax = 30.
t = np.linspace(0.,tmax, Nt)


#Plotting
fig = plt.figure()
#plt.title("Lotka-Volterra")
#plt.xlabel('Time')
#plt.ylabel('Population')
plt.ylim(-0.15,1.15)
plt.xlim(-1,31)
plt.axis('off')


curve, = plt.plot([],[],'k-', linewidth=3)
writer = PillowWriter(fps = 15)
with writer.saving(fig, "lotka.gif", 100):
    for b in np.linspace(0,1,100):
        res = solve(r,k,alpha,b*a,P0,t)
        curve.set_data(t,res.T[-1])
        writer.grab_frame()
