import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
import itertools

"""
Implementing cellular automata that are variations of the Game of Life.

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
    def __init__(self, C = np.random.rand(50, 50) > .5, rule = "B3S23"):
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

    def show_grid(self):
        plt.figure()
        plt.imshow(self.C, cmap = cm.gray)
        plt.axis('off')
        plt.show()

#a function for making gif animations of iterations of the automata
def makegif(g, t):
    def updatefig(*args):
        g.iterate()
        im.set_array(g.C)
        return im,

    fig = plt.figure()
    plt.axis('off')
    im = plt.imshow(g.C, interpolation = "nearest", cmap = cm.binary)
    writer = animation.PillowWriter(fps = 15)
    with writer.saving(fig, "automato.gif", 100):
        for n in range(t):
            updatefig()
            writer.grab_frame()
            writer.grab_frame()

#creating a random N-sized grid for testing the Game of Life
def maketest(N = 100):
    t = np.random.rand(N + 1)
    X, Y = np.meshgrid(t, t)
    f = 4
    C0 = np.sin(2. * np.pi * f * X ) * np.sin(2. * np.pi * 2 * f * Y )  > -.1
    g = LLCA(C0, rule = "B2S23")
    return g


#seeking for interesting alternative rules to the Game of Life by looking for repeating patterns
def cartesian(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def seeker(batch = 100, iteractions = 100):
    births = cartesian(np.arange(4),np.arange(4))
    survivals = cartesian(np.arange(4),np.arange(4))

    possible_two_rules = []
    for a,b in zip(births,survivals):
        rule = (a,b)
        possible_two_rules.append(rule)

    possible_cool_list = []
    bunch = [np.random.rand(5, 5) > .5 for _ in range(batch)]
    for rule in possible_two_rules:
        for b in bunch:
            seen_patterns = []
            moo = LLCA(b, rule = "B" + str(rule[0][0]) + str(rule[0][1]) + "S" + str(rule[1][0]) + str(rule[1][1]))
            for _ in range(iteractions):
                moo.iterate()
                if moo.C in seen_patterns:
                    soo = LLCA(moo.C, moo.rule)
                    possible_cool_list.append(soo)
                    break
                else:
                    seen_patterns.append(moo.C)

    return possible_cool_list
