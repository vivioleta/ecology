import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from matplotlib.animation import PillowWriter

"""

"""
class model:
        def __init__(self,  A, V, S, rs = 65.0, ra = 2.0, h = 0.5, c = 1.05, pa = 2.0, ps = 1.0, e = 0.3, delta0 = 0.5, deltaS = 1.5, dn = 1.0, dsucesso = 1.0):
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
            self.dn = dn
            self.dsucesso = dsucesso

            self.N = A.shape[0]
            self.LCM = np.array([np.sum(np.delete(A,i,0),axis = 0) / (self.N - 1) for i in range(self.N)])
            self.GCM = np.sum(A,axis = 0)/self.N
            self.goal = np.linalg.norm(self.GCM) <= dsucesso

        def distance(P,Q):
            return np.linalg.norm(P-Q)

        def normalize(P):
            norm = np.linalg.norm(P)
            return P/norm

        def updateAgent(self, i):
            epsilon = model.normalize(np.random.normal(size = 2)) #choosing a random unit vector
            if model.distance(self.A[i],self.S) > self.rs:
                self.A[i] = self.A[i] + self.e*epsilon
                return self
            else:
                closeneighbors = np.array([self.A[j] for j in range(self.N) if (model.distance(self.A[i],self.A[j]) < self.ra) & (j != i)])
                if np.any(closeneighbors):
                     repels = np.sum(self.A[i] - closeneighbors, axis = 0)
                else:
                    repels = np.zeros(2)
                self.V[i] = (self.h)*self.V[i] + (self.c) * (self.LCM[i] - self.A[i]) + (self.pa)*repels + (self.ps)*(self.A[i] - self.S) + (self.e)*epsilon
                self.A[i] = self.A[i] + (self.delta0)*(self.V[i])
                return self

        def updateShepherd(self):
            distances_gcm = np.array([model.distance(a,self.GCM) for a in self.A])
            distances_shepherd = np.array([model.distance(a,self.S) for a in self.A])
            #making the shepherd chill if they're too close to an agent
            if any(distances_shepherd < (3*self.ra)):
                return self
            #collecting
            if any(distances_gcm > self.dn):
                runaway_index = np.argmax(distances_gcm)
                runaway = self.A[runaway_index]
                Pc = runaway + (3*self.ra) * model.normalize(runaway - self.GCM)
                self.S = self.S + self.deltaS * model.normalize(Pc - self.S)
                return self
            #herding
            else:
                Pd = self.GCM + self.ra*sqrt(self.N)*model.normalize(self.GCM)
                self.S = self.S + self.deltaS * model.normalize(Pd - self.S)
                return self
        def updateAll(self):
            for i in range(self.N): self.updateAgent(i)
            self.updateShepherd()
            return self

#Testing and making animations
L = 150
n = 15
x = L*np.random.rand(n)
y = L*np.random.rand(n)
coord = np.column_stack((x,y))
xS = L*np.random.rand(1)
yS = L*np.random.rand(1)
coordS = np.column_stack((xS,yS))

aff = np.array(coord)
bff = np.zeros([2,n]).T
cff = np.array(coordS)
moo = model(aff,bff,cff)


fig = plt.figure()
plt.ylim(-20,2*L)
plt.xlim(-20,2*L)


agents, = plt.plot(moo.A.T[0],moo.A.T[1],'xb')
shepherd, = plt.plot(moo.S.T[0],moo.S.T[1],'or')
plt.show()

writer = PillowWriter(fps = 15)
with writer.saving(fig, "herding.gif", 100):
    for b in np.linspace(0,1,500):
        moo.updateAll()
        agents.set_data(moo.A.T[0],moo.A.T[1])
        shepherd.set_data(moo.S.T[0],moo.S.T[1])
        writer.grab_frame()
