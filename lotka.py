import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from scipy import integrate
import random

"""
Implementing some versions of the Lotka-Volterra equations for populations modeling.


"""

#Lotka-Volterra equations for a pair of predator-prey species
def lotka_prey(P, t, alpha, beta, delta, gamma):
    p, q = P
    dotp = p * (alpha - beta * q)
    dotq = q * (-delta + gamma * p)
    return np.array([dotp, dotq])

def solve_prey(P0, t, alpha, beta, delta, gamma):
    res = integrate.odeint(lotka_prey, P0, t, args = (alpha, beta, delta, gamma))
    return res


#Lotka-Volterra equations for n competing species
def lotka_competitive(P, t, r, k, alpha):
    dotP = r*P - (r/k)*P*(alpha@P)
    return dotP

def solve_competitive(P0,t,r,k,alpha):
    res = integrate.odeint(lotka_competitive, P0, t, args = (r,k,alpha))
    return res


#Lotka-Volterra equations for n competing species with Allee effects
def lotka_allee(P, t, r, k, a, alpha):
    np.fill_diagonal(alpha,P - a) #using this term for the diagonals to simplify the expression for the Lotka-Volterra equations
    dotP = r*P*(P - a) - (r/k)*P*(alpha@P)
    return dotP

def solve_Allee(P0,t,r,k,a,alpha):
    res = integrate.odeint(lotka_allee, P0, t, args = (r,k,a,alpha))
    return res


#Creating a test for Allee and making a gif variating Allee parameters
def maketest_Allee():
    n = 4
    r = np.array([1,2,3,4]) #vector with unconstrained growh rates for each species
    k = np.ones(n) #vector with the carrying campacities for each species
    a = np.ones(n)
    alpha = np.array([[1,0.4,0.7,0.8],[0.1,1,2,0.04],[0.3,0.5,1,0.9],[0.1,0.1,0.1,1]]) #matrix with the parameters for each possible species interaction, with alpha_ii = 1 for all i
    P0 = np.ones(n) #initial values

    Nt = 1000
    tmax = 30.
    t = np.linspace(0.,tmax, Nt)
    
    return r,k,a,alpha,P0,t

def makegif_Allee():
    r,k,a,alpha,P0,t = maketest_Allee()

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
            res = solve_Allee(P0,t, r,k,b*a,alpha)
            curve.set_data(t,res.T[-1])
            writer.grab_frame()
