import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation, cm

"""
Implemeting some cellular automata.

Notes:
(1): Code for the Game Of Life automata mostly taken from https://scientific-python.readthedocs.io/en/latest/notebooks_rst/0_Python/10_Examples/GameOfLife.html

"""

class LLCA:
    """
    A Life Like Cellular Automaton (LLCA)

    Inputs:
    * C: a binary matrix representing the cells where 1 stands for alive and 0 for dead.
    * rule: the rule of the in the format 'BXSY' where X and Y are the birth and survival conditions.
            Example: GOL rule is "B3S23".
    """
    def __init__(self, C = np.random.rand(50, 50), rule = "B3S23"):
        self.C = np.array(C).astype(np.bool_)
        self.rule = rule

    def parse_rule(self):
        """
        Parses the rule string
        """
        r = self.rule.upper().split("S")
        B = np.array([int(i) for i in r[0][1:] ]).astype(np.int64)
        S = np.array([int(i) for i in r[1] ]).astype(np.int64)
        return B, S

    def neighbors(self):
        """
        Returns the number of living neigbors of each cell.
        """
        C = self.C
        N = np.zeros(C.shape, dtype = np.int8) # Neighbors matrix
        N[ :-1,  :  ]  += C[1:  , :  ] # Living cells south
        N[ :  ,  :-1]  += C[ :  ,1:  ] # Living cells east
        N[1:  ,  :  ]  += C[ :-1, :  ] # Living cells north
        N[ :  , 1:  ]  += C[ :  , :-1] # Living cells west
        N[ :-1,  :-1]  += C[1:  ,1:  ] # Living cells south east
        N[1:  ,  :-1]  += C[ :-1,1:  ] # Living cells north east
        N[1:  , 1:  ]  += C[ :-1, :-1] # Living cells north west
        N[ :-1, 1:  ]  += C[1:  , :-1] # Living cells south west
        return N

    def iterate(self):
        """
        Iterates one time.
        """
        B, S = self.parse_rule()
        N = self.neighbors()
        C = self.C
        C1 = np.zeros(C.shape, dtype = np.int8)
        for b in B: C1 += ((C == False) & (N == b))
        for s in S: C1 += (C & (N == s))
        self.C[:] = C1 > 0

def make_test():
    N = 100
    t = np.random.rand(N + 1)
    X, Y = np.meshgrid(t, t)
    f = 4
    C0 = np.sin(2. * np.pi * f * X ) * np.sin(2. * np.pi * 2 * f * Y )  > -.1
    g = LLCA(C0, rule = "B2S23")
    return g

def show_grid(g):
    plt.figure()
    plt.imshow(g.C, cmap = cm.gray)
    plt.axis('off')
    plt.show()


def make_gif(g):
    def updatefig(*args):
        g.iterate()
        im.set_array(g.C)
        return im,

    fig = plt.figure()
    plt.axis('off')
    im = plt.imshow(g.C, interpolation = "nearest", cmap = cm.binary)
    writer = animation.PillowWriter(fps = 15)
    with writer.saving(fig, "automato.gif", 100):
        for n in range(100):
            updatefig()
            writer.grab_frame()
            writer.grab_frame()
