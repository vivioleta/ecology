import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from scipy import integrate
import random

"""
Implementing some versions of the Lotka-Volterra equations for populations modeling.

Notes:

(1) The general Lotka-Veltorra equations can be used to model the three equations here (predator prey, competitive and competitive with Allee effects) but for the Allee case you need to let the r and A parameters contain non constant terms. Due to the way I coded this, I had to do aan ugly little condition case for the Allee equation.


"""



class general_lotka_volterra:

    def __init__(self, r, A, a = np.array([])):
        self.r = r      #unconstrained growh vector
        self.A = A      #community matrix
        self.a = a      #allee population effect

        self.N = self.r.shape[0]

    def derivative(t, P, r, A, a = np.array([])):
        if a.any():
            I = np.zeros(A.shape)
            np.fill_diagonal(I,P - a)
            A = I@A
            dotP = P*(r*(P - a) + A@P)
        else:
            dotP = P*(r + A@P)       #The General Lotka-Volterra equation
        return dotP

    def solve(time_interval, initialPop, r, A, a = np.array([])):
        sol = integrate.solve_ivp(general_lotka_volterra.derivative, time_interval, initialPop, args = (r, A, a))
        return sol

    def makePlot(self, time_interval, initialPop):
        sol = general_lotka_volterra.solve(time_interval,initialPop, self.r, self.A, self.a)
        fig = plt.figure()
        for i in range(self.N): plt.plot(sol.t,sol.y[i], label = str(i))
        return fig

    def makeOnePlot(self, time_interval, initialPop, i):
        sol = general_lotka_volterra.solve(time_interval,initialPop, self.r, self.A, self.a)
        fig = plt.figure()
        plt.plot(sol.t,sol.y[i], label = str(i))
        return fig


#The Lotka-Volterra predator prey pair equations
def prey_predator(alpha, beta, gamma, delta):
    r = np.array([alpha,-gamma])
    A = np.array([[0,-beta],[delta,0]])
    model = general_lotka_volterra(r,A)
    return model

def make_prey_example(t = 100):
    alpha = 5*np.random.random()
    beta = 5*np.random.random()
    gamma = 5*np.random.random()
    delta = 5*np.random.random()
    moo = prey_predator(alpha,beta,gamma,delta)

    p0 = 10*np.random.random()
    q0 = 10*np.random.random()
    initialPop = np.array([p0,q0])

    time = np.array([0,t])
    fig = moo.makePlot(time,initialPop)
    plt.legend()
    plt.show()

#Lotka-Volterra equations for n competing species
def lotka_competitive(r, k, alpha):
    A = (r/k)*alpha
    model = general_lotka_volterra(r,A)
    return model

def make_competitive_example(t = 100):
    r = np.array([5*np.random.random(),5*np.random.random()])
    k = np.array([100*np.random.random(),100*np.random.random()])
    alpha = np.array([[-1,-5*np.random.random()],[-5*np.random.random(),-1]])
    moo = lotka_competitive(r, k, alpha)

    p0 = 10*np.random.random()
    q0 = 10*np.random.random()
    initialPop = np.array([p0,q0])

    time = np.array([0,t])
    fig = moo.makePlot(time,initialPop)
    sol = general_lotka_volterra.solve(time,initialPop, moo.r, moo.A)
    plt.legend()
    plt.show()
    return moo

#Lotka-Volterra equations for n competing species with Allee effects
def lotka_allee(r, k, a, alpha):
    A = (r/k)*alpha
    model = general_lotka_volterra(r,A,a)
    return model

def make_allee_example(t = 100):
    r = np.array([5*np.random.random(),5*np.random.random()])
    k = np.array([100*np.random.random(),100*np.random.random()])
    a = np.array([np.random.random(),np.random.random()])
    alpha = np.array([[-1,-5*np.random.random()],[-5*np.random.random(),-1]])
    moo = lotka_allee(r, k, a, alpha)

    p0 = 10*np.random.random()
    q0 = 10*np.random.random()
    initialPop = np.array([p0,q0])

    time = np.array([0,t])
    fig = moo.makePlot(time,initialPop)
    sol = general_lotka_volterra.solve(time,initialPop, moo.r, moo.A)
    plt.legend()
    plt.show()
    return moo


#Making pretty gifs: variating allee parameter, chaotic dynamics for competitive with N = 4, and oscillating for predator-prey












"""
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

"""
