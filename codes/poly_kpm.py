from itertools import product
from functools import reduce
import numpy as np



class SumOfOperatorProducts:
    def __init__(self, terms):
        self.terms = terms # [[(array, "AB"), (array, "BB")], [(array, "AB")]]
        self.simplify_products()

    def __add__(self, other):
        # Actually if AB should add things together;
        return SumOfOperatorProducts(self.terms + other.terms)
    
    def __neg__(self):
        temp = []
        for sublist in self.terms:
            sublist[0] = ((-1)*sublist[0][0],sublist[0][1])
            temp.append(sublist)
        return SumOfOperatorProducts(temp)
    
    def __matmul__(self, other):
        return SumOfOperatorProducts([a + b for a, b in product(self.terms, other.terms)])
    
    def __truediv__(self, other): # self / other
        return (1/other) * self
    
    def __mul__(self, other):
        temp = []
        for sublist in self.terms:
            sublist[0] = (other*sublist[0][0],sublist[0][1])
            temp.append(sublist)
        return SumOfOperatorProducts(temp)
    
    def __rmul__(self,other):
        return self * other
    
    def conjugate(self):
        temp = [[(v[0].conjugate(),v[1]) for v in slist] for slist in self.terms]
        return SumOfOperatorProducts(temp)
    
    def transpose(self):
        temp = [[(v[0].transpose(),v[1]) for v in slist] for slist in self.terms]
        return SumOfOperatorProducts(temp)
    
    def reduce_sublist(self,slist, c_flag='B'):
        # This can be made more efficient by getting rid of the surplus loop
        # to check equality
        def elmmul(a,b):
            return (a[0]@b[0], '{0}{1}'.format(a[1][0],b[1][1]))
        
        temp = [slist[0]]
        for v in slist[1:]:
            if temp[-1][1][1] == v[1][0] and v[1][0] == c_flag:
                    temp[-1] = elmmul(temp[-1],v)
            else:
                temp.append(v)
        if len(temp)<len(slist):
            return self.reduce_sublist(temp)
        elif len(temp)==len(slist):
            return temp
    
    def simplify_products(self):
        nterms = [self.reduce_sublist(slist) for slist in self.terms]
        self.terms = nterms
        return
    
    def sum_sublist(self,slist,flag):
        return sum([v[0] for v in slist if v[1]==flag])
        
    def evalf(self,flag):
        temp = [self.reduce_sublist(slist, c_flag='A') for slist in self.terms]
        
        if flag is None:
            flags = list(str(v[0]+v[1]) for v in product(['A','B'],['A','B']))
        else:
            flags = [flag]

        sec_temp = []
        for flag in flags:
            sec_temp.append((sum(map(lambda x:self.sum_sublist(x,flag),temp)),
                             str(flag)))
         
        return sec_temp

def divide_energies(Y, H_0_AA, H_0_BB):
    E_A = np.diag(H_0_AA.evalf('AA')[0][0])
    E_B = np.diag(H_0_BB.evalf('BB')[0][0])
    energy_denoms = 1/(E_A.reshape(-1, 1) - E_B)
    
    return Y * energy_denoms

# +
from numpy.random import random as rnd
t_list = [[(rnd((4,10)),'AB'),
           (rnd((10,10)),'BB'),
           (rnd((10,4)),'BA'),
           (rnd((4,10)),'AB')],
          [(rnd((4,10)),'AB'),
           (rnd((10,10)),'BB')],
          [(rnd((4,4)),'AA'),
           (rnd((4,10)),'AB'),
           (rnd((10,4)),'BA'),
           (rnd((4,10)),'AB'),
           (rnd((10,10)),'BB'),
           (rnd((10,4)),'BA'),
           (rnd((4,4)),'AA'),
           (rnd((4,10)),'AB')]]

t_list_2 = [[(rnd((4,10)),'AB'),
             (rnd((10,10)),'BB')],
            [(rnd((10,4)),'BA'),
             (rnd((4,4)),'AA'),
             (rnd((4,10)),'AB')]]

# -


