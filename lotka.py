import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import integrate
import random
import matplotlib
from scipy.interpolate import griddata

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
            dotP = np.zeros(r.shape)
            for i in range(r.shape[0]): dotP[i] = P[i]*(r[i]*(P - a)[i] + (A@P)[i])
        else:
            #The General Lotka-Volterra equation:    dotP[i] = P[i]*(r[i] + A@P[i])
            dotP = np.zeros(r.shape)
            for i in range(r.shape[0]): dotP[i] = P[i]*(r[i] + (A@P)[i])
        return dotP

    def solve(time_interval, initialPop, r, A, a = np.array([])):
        sol = integrate.solve_ivp(general_lotka_volterra.derivative, time_interval, initialPop, args = (r, A, a))
        return sol

    def makePlot(self, time_interval, initialPop):
        sol = general_lotka_volterra.solve(time_interval,initialPop, self.r, self.A, self.a)
        fig = plt.figure()
        for i in range(self.N): plt.plot(sol.t,sol.y[i], label = "Species " + str(i))
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

    p0 = np.random.random()
    q0 = np.random.random()
    initialPop = np.array([p0,q0])

    time = np.array([0,t])
    fig = moo.makePlot(time,initialPop)
    plt.legend()
    plt.title("Lotka-Volterra equations for a predator pray pair")
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.show()

def make_cool_prey_gif(t = 40):
    #showing the time evolution of the populations
    alpha = 5*np.random.random()
    beta = 5*np.random.random()
    gamma = 5*np.random.random()
    delta = 5*np.random.random()
    moo = prey_predator(alpha,beta,gamma,delta)

    p0 = np.random.random()
    q0 = np.random.random()
    initialPop = np.array([p0,q0])
    time = np.array([0,t])
    s = general_lotka_volterra.solve(time, initialPop, moo.r, moo.A)

    fig = plt.figure()
    y1max = s.y[0][np.argmax(s.y[0])]
    y2max = s.y[1][np.argmax(s.y[1])]
    plt.xlim(0, t)
    plt.ylim(0, max(y1max,y2max))
    plt.axis('off')

    firstpop, = plt.plot([], [],)
    secondpop, = plt.plot([], [])

    point1List = []
    point2List = []

    writer = PillowWriter(fps = 15)
    with writer.saving(fig, "prey.gif", 100):
        for i in range(s.t.shape[0]):
            point1List.append(s.y[0][i])
            point2List.append(s.y[1][i])
            firstpop.set_data(s.t[:i+1],point1List)
            secondpop.set_data(s.t[:i+1],point2List)
            writer.grab_frame()

#Lotka-Volterra equations for n competing species
def lotka_competitive(r, k, alpha):
    A = np.zeros(alpha.shape)
    for i in range(r.shape[0]): A[i] = (-r/k)[i]*alpha[i]
    model = general_lotka_volterra(r,A)
    return model

def make_competitive_example(t = 300):
    #parameters taken from Wikipedia
    #here I'm plotting an equilibrium point where all four species co-exist stably in the same density. Due to sensitivity of initial conditions, the dynamics diverge from equilibrium after some time, showing the chaotic behaviour of this choice of parameters
    r = np.array([1,.72,1.53,1.27])
    k = np.ones(4)
    alpha = np.array([[1,1.09,1.52,0],[0,1,.44,1.36],[2.33,0,1,.47],[1.21,.51,.35,1]])
    moo = lotka_competitive(r,k,alpha)

    initialPop = np.array([0.3013,0.4586,0.1307,0.3557])
    time = np.array([0,t])


    fig = moo.makePlot(time,initialPop)
    plt.title("Lotka-Volterra equations for 4 competing species with chaotic dynamics - equilibrium point")
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.show()
    return moo

def make_competitive_gif(t = 35):
    #parameters taken from Wikipedia, lowest dimensional example where chaotic dynamics happen
    #variating the initial population
    r = np.array([1,.72,1.53,1.27])
    k = np.ones(4)
    alpha = np.array([[1,1.09,1.52,0],[0,1,.44,1.36],[2.33,0,1,.47],[1.21,.51,.35, 1]])
    moo = lotka_competitive(r,k,alpha)

    time = np.array([0,t])

    fig = plt.figure()
    plt.ylim(0,1)
    plt.xlim(0,t)
    plt.axis('off')

    curve0, = plt.plot([],[],'red', linewidth=3)
    curve1, = plt.plot([],[],'green', linewidth=3)
    curve2, = plt.plot([],[],'blue', linewidth=3)
    curve3, = plt.plot([],[],'yellow', linewidth=3)

    writer = PillowWriter(fps = 15)
    with writer.saving(fig, "chaos.gif", 100):
        for x in np.linspace(0,1,200):
            initialPop = np.array([0.232*x,0.5,0.1134*x,0.823])
            sol = general_lotka_volterra.solve(time,initialPop,moo.r,moo.A)
            curve0.set_data(sol.t,sol.y[0],)
            curve1.set_data(sol.t,sol.y[1])
            curve2.set_data(sol.t,sol.y[2])
            curve3.set_data(sol.t,sol.y[3])
            writer.grab_frame()

#Lotka-Volterra equations for n competing species with Allee effects
def lotka_allee(r, k, a, alpha):
    A = np.zeros(alpha.shape)
    for i in range(r.shape[0]): A[i] = (-r/k)[i]*alpha[i]
    model = general_lotka_volterra(r,A,a)
    return model

def make_allee_example(t = 100):
    r = np.array([5*np.random.random(),5*np.random.random()])
    k = np.ones(2)
    a = np.array([np.random.random(),np.random.random()])
    alpha = np.array([[1,5*np.random.random()],[5*np.random.random(),1]])
    moo = lotka_allee(r, k, a, alpha)

    p0 = np.random.random()
    q0 = np.random.random()
    initialPop = np.array([p0,q0])

    time = np.array([0,t])
    fig = moo.makePlot(time,initialPop)
    plt.title("Lotka-Volterra equations with Allee conditions")
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.show()
    return moo

def make_cool_allee_gif(t = 50):
    #Variating the Allee parameters
    r = np.array([5*np.random.random(),5*np.random.random()])
    k = np.ones(2)
    a = np.array([np.random.random(),np.random.random()])
    alpha = np.array([[1,5*np.random.random()],[5*np.random.random(),1]])
    p0 = np.random.random()
    q0 = np.random.random()
    initialPop = np.array([p0,q0])
    time = np.array([0,t])

    fig = plt.figure()
    plt.ylim(0,1)
    plt.xlim(0,t)
    plt.axis('off')

    curve, = plt.plot([],[],'k-', linewidth=3)
    writer = PillowWriter(fps = 15)
    with writer.saving(fig, "allee.gif", 100):
        for b in np.linspace(0,1,150):
            moo = lotka_allee(r,k,b*a,alpha)
            sol = general_lotka_volterra.solve(time,initialPop,moo.r,moo.A,moo.a)
            curve.set_data(sol.t,sol.y[-1])
            writer.grab_frame()
