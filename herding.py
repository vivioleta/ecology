import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from matplotlib.animation import PillowWriter

"""

Implemeting the herding model from https://royalsocietypublishing.org/doi/full/10.1098/rsif.2014.0719

"""

class model:
        def __init__(self,  A, V, S, rs = 65.0, ra = 5.0, h = 0.5, c = 1.05, pa = 2, ps = 1.0, e = 0.3, delta0 = 1, deltaS = 4.5):
            self.A= np.float64(A)
            self.V = np.float64(V)
            self.S= np.float64(S)

            self.rs = rs
            self.ra = ra
            self.h = h
            self.c = c
            self.pa = pa
            self.ps = ps
            self.e = e
            self.delta0 = delta0
            self.deltaS = deltaS

            self.N = self.A.shape[0]
            self.LCM = np.array([np.sum(np.delete(self.A,i,0),axis = 0) / (self.N - 1) for i in range(self.N)])
            self.GCM = np.sum(self.A,axis = 0)/self.N
            self.dn = self.ra*(self.N)**(2/3)
            self.goal = np.linalg.norm(self.GCM) <= self.dn

        def distance(P,Q):
            return np.linalg.norm(P-Q)

        def normalize(P):
            norma = np.linalg.norm(P)
            if norma == 0:  return P
            return P/norma

        def updateAgent(self, i):
            epsilon = model.normalize(np.random.normal(size = 2)) #choosing a random unit vector
            if model.distance(self.A[i],self.S) > self.rs:
                self.A[i] = self.A[i] + self.e*epsilon
                return model(self.A,self.V,self.S)
            else:
                closeneighbors = np.array([self.A[j] for j in range(self.N) if (model.distance(self.A[i],self.A[j]) < self.ra) & (j != i)])
                if np.any(closeneighbors):
                    vetorzinho = np.array([model.normalize(self.A[i] - b) for b in closeneighbors])
                    repels = model.normalize(np.sum(vetorzinho, axis = 0))
                else:
                    repels = np.zeros(2)
                self.V[i] = (self.h)*self.V[i] + (self.c) * model.normalize((self.LCM[i] - self.A[i])) + (self.pa)*repels + (self.ps)*model.normalize(self.A[i] - self.S) + (self.e)*epsilon
                self.A[i] = self.A[i] + (self.delta0)*self.V[i]
                return model(self.A,self.V,self.S)

        def updateShepherd(self):
            distances_gcm = np.array([model.distance(a,self.GCM) for a in self.A])
            distances_shepherd = np.array([model.distance(a,self.S) for a in self.A])
            #making the shepherd chill if they're too close to an agent
            if any(distances_shepherd < (3*self.ra)):
                return model(self.A,self.V,self.S)
            #collecting
            if np.any(distances_gcm > self.dn):
                print('Collecting')
                runaway_index = np.argmax(distances_gcm)
                runaway = self.A[runaway_index]
                Pc = runaway + (3*self.ra) * model.normalize(runaway - self.GCM)
                self.S = self.S + self.deltaS * model.normalize(Pc - self.S)
                return model(self.A,self.V,self.S)
            #herding
            else:
                print('Herding')
                Pd = self.GCM + self.ra*sqrt(self.N)*model.normalize(self.GCM)
                self.S = self.S + self.deltaS * model.normalize(Pd - self.S)
                return model(self.A,self.V,self.S)
        def updateAll(self):
            for i in range(self.N): self = self.updateAgent(i)
            self = self.updateShepherd()
            if self.goal: print ("Successful shepherding!")
            return self

#Testing and making animations
L = 150
n = 10
x = L*np.random.rand(n)
y = L*np.random.rand(n)
coord = np.column_stack((x,y))
xS = L*np.random.rand(1)
yS = L*np.random.rand(1)
coordS = np.column_stack((xS,yS))

aff = np.array(coord)
bff = np.zeros([2,n]).T + 0.001
cff = np.array(coordS)
moo = model(aff,bff,cff)


fig = plt.figure()
plt.ylim(-2*L,2*L)
plt.xlim(-2*L,2*L)


agents, = plt.plot(moo.A.T[0],moo.A.T[1],'xb')
shepherd, = plt.plot(moo.S.T[0],moo.S.T[1],'or')
showgcm, = plt.plot(moo.GCM.T[0],moo.GCM.T[1], 'og')

writer = PillowWriter(fps = 15)
with writer.saving(fig, "herding.gif", 100):
    for b in np.linspace(0,1,600):
        moo = moo.updateAll()
        agents.set_data(moo.A.T[0],moo.A.T[1])
        shepherd.set_data(moo.S.T[0],moo.S.T[1])
        showgcm.set_data(moo.GCM.T[0],moo.GCM.T[1])
        writer.grab_frame()
