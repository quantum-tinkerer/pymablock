import sympy
import numpy as np
import tinyarray as ta

def sym_to_ta(h):
    """
    INPUTS:
    h           : dict
                dict containing the H_0 hamiltonian with key '1' and all pertubing hamiltonians 
                along their respective symbols
                
    OUTPUT:
    hpn:        : dict
                dict with the original keys of h replaced by ta.arrays
    all_keys    : list of all symbols contained in h with order representing the map 
                on ta.array entries
    """
    all_keys = list(sum(h.keys()).free_symbols)
    # generate keys
    hpn = {}
    for k, v in h.items():
        # generate key array
        nkey = np.zeros_like(all_keys)

        if isinstance(k,sympy.core.symbol.Symbol):
            nkey[all_keys.index(k)] = 1

        if isinstance(k,sympy.core.power.Pow):
            nkey[all_keys.index(k.base)] = k.exp
            
        if isinstance(k,sympy.core.mul.Mul):
            for sub in k.args:
                if isinstance(sub, sympy.core.symbol.Symbol):
                    nkey[all_keys.index(k)] = 1
                if isinstance(sub, sympy.core.power.Pow):
                    nkey[all_keys.index(k.base)] = sub.exp
                    
        hpn[ta.array(nkey)] = v
        
    return hpn, all_keys


def ta_to_symb(hp,keys):
    """
    IPUTS:
    hp          : dict 
                perturbed Hamiltonian with ta.arrays as keys
    keys        : list
                List of sympy symbols encoding the entries of the tinyarrays
                
    OUTPUT:
    hpn         : dict
                dictionary containing hamiltonian with perturbation symbols as keys
    """
    hpn = {}
    for k,v in hp.items():
        n_key = sympy.prod([keys[i]**k[i] for i in range(len(k))])
        hpn[n_key] = v
    return hpn