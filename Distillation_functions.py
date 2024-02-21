import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
import mpmath
import sympy as sp
import pandas as pd
import pickle
import sys
import math
import os
mpmath.mp.dps = 80 
# getcontext().prec = 70  


# Load all the final_prob_dict dictionaries into a dictionary
max_rep_code = 12
n_values = np.linspace(2,max_rep_code,max_rep_code-1, dtype=int)
final_prob_dicts = {}
for n in n_values:
    with open(f'repetition_code_prob_dict__n_{n}.pkl', 'rb') as file:
        final_prob_dicts[n] = pickle.load(file)


def ED_C_n_1_n(n, p, printing=False): # [n,1,n] repetition code
    final_prob_dict = final_prob_dicts[n]
    
    pI = mpmath.mpf(p[0])
    pX = mpmath.mpf(p[1])
    pZ = mpmath.mpf(p[2])
    pY = mpmath.mpf(p[3])
    
    # Here put the values of pI, pX, pZ, pY so that LpI, LpX, LpZ, LpY will be numbers and not functions:
    LpI_expr = final_prob_dict['IL']
    LpX_expr = final_prob_dict['XL']
    LpZ_expr = final_prob_dict['ZL']
    LpY_expr = final_prob_dict['YL']
    
    # Substitute the values of pI, pX, pZ, pY into the expressions
    LpI = LpI_expr.subs({'pI': pI, 'pX': pX, 'pZ': pZ, 'pY': pY})
    LpX = LpX_expr.subs({'pI': pI, 'pX': pX, 'pZ': pZ, 'pY': pY})
    LpZ = LpZ_expr.subs({'pI': pI, 'pX': pX, 'pZ': pZ, 'pY': pY})
    LpY = LpY_expr.subs({'pI': pI, 'pX': pX, 'pZ': pZ, 'pY': pY})
        
    norm = LpI + LpX + LpZ + LpY
    p_reject = mpmath.mpf(1) - norm # rejection probability
    rate = (mpmath.mpf(1) / mpmath.mpf(n)) * (mpmath.mpf(1) - p_reject)
    if printing:
        print(f"probability of success in [2,1,2] step = {1-float(p_reject):.2e}") # for test - remove!
    return rate, [LpI/norm, LpX/norm, LpZ/norm, LpY/norm]



def depolarizing(p): # from a scalar p to a vector pI, pX, pZ, pY
    if isinstance(p, mpmath.mpf):
        return [mpmath.mpf(1) - p, p / mpmath.mpf(3), p / mpmath.mpf(3), p / mpmath.mpf(3)]
    elif isinstance(p, list):
        if len(p) == 1:
            return [mpmath.mpf(1) - p[0], p[0] / mpmath.mpf(3), p[0] / mpmath.mpf(3), p[0] / mpmath.mpf(3)]
        elif len(p) > 1:
            return p
    raise ValueError("Invalid input. Expected an mpf number or a list.")

def hadamard(p): # I,X,Z,Y --> I,Z,X,Y
    return [p[0], p[2], p[1], p[3]] 

def s_mat(p): # I,X,Z,Y --> I,Y,Z,X
    # we use HSH to take (I,X,Z,Y)->(I,X,Y,Z)
    return [p[0], p[3], p[2], p[1]] 


def infidelity(p):
    if isinstance(p, mpmath.mpf):
        return p
    elif isinstance(p, list) and len(p) >= 4:
        return mpmath.mpf(p[1] + p[2] + p[3])
    else:
        raise ValueError("Invalid input. Expected an mpf number or a list with at least four elements.")


def calculate_complement_sum(n, p, d):
    complement_sum = mpmath.mpf(0)
    for i in range(d):
        binomial_coeff = mpmath.binomial(n, i)
        term = binomial_coeff * (p ** i) * ((1 - p) ** (n - i))
        complement_sum += term
    return 1 - complement_sum


def ED_n_1_n(n, in_error, basis = 'Z', printing=False): # Classical repetition codes, for different bases. Out error is a vector
    # Change basis:
    if basis == 'X':
        in_error = hadamard(depolarizing(in_error))
    elif basis == 'Y':
        in_error = hadamard(s_mat(hadamard(depolarizing(in_error))))
    
    # Repetition code in Z basis:
    eff_rate, out_error = ED_C_n_1_n(n, depolarizing(in_error), printing=printing)

    # Change basis again:
    if basis == 'X':
        out_error = hadamard(out_error)
    elif basis == 'Y':
        out_error = hadamard(s_mat(hadamard(out_error)))
        
    out_qubits = 1
    return eff_rate, out_error, out_qubits


def ED_n_k_d(in_error, n, k, d=2, printing=False): # Error detection with any code [n,k,d]
    out_error = calculate_complement_sum(n, in_error, d) / ((1 - in_error) ** n)
    # out_error = calculate_complement_sum(n, p=in_error, d=d)
    eff_rate = (k / n) * (mpmath.mpf(1) - in_error) ** n
    out_qubits = k
    if printing:
        print(f"probability of success in [[n,k,d]] step = { float((mpmath.mpf(1) - in_error) ** n):.2e}") # for test - remove!
    return eff_rate, out_error, out_qubits


class Parameters:
    def __init__(self, max_memory, target_output_error, rate_seq, total_memory_seq, K_seq) -> None:
        self.max_memory = max_memory
        self.target_output_error = target_output_error
        self.rate_seq = rate_seq
        self.total_memory_seq = total_memory_seq
        self.K_seq = K_seq
        
        
class Sequence_results:
    def __init__(self, code_sequence, output_error, total_memory_seq, overhead, K_seq, memory_sequence) -> None:
        self.code_sequence = code_sequence
        self.output_error = output_error
        self.total_memory_seq = total_memory_seq
        self.overhead = overhead
        self.K_seq = K_seq
        self.memory_sequence = memory_sequence
        self.memory_usage = max(memory_sequence)
        
    def __str__(self):
        print_code_sequence = [[code[0], code[1], code[2], code[3], code[4]] if code[3]=='Classical' else [code[0], code[1], code[2], code[3]] for code in self.code_sequence]
        return f"code_sequence={print_code_sequence}, \
        infidelity per qubit={float(self.output_error / self.K_seq):.1e}, memory_sequence = {self.memory_sequence}, memory_usage={float(self.memory_usage):.1e}, overhead per qubit ={float(self.overhead):.3e}"


def evaluate_sequence(code, in_error, max_memory, target_output_error, rate_seq, total_memory_seq,  K_seq, prev_memory_usage, printing=False):
    # Distillation step l
    n,k,d, type = code[0], code[1], code[2], code[3] # [n,k,d] of this level
    # Classical codes - return fidelity vector:
    if type == 'Classical':
        if n == d: # repetition code:
            eff_rate, out_error, out_qubits = ED_n_1_n(n, in_error=in_error, basis = code[4], printing=printing)
        else:
            return 'failure', 0, 0, 0, 0, 0, 0, 0
    # Quantum codes - return fidelity scalar
    elif type == 'Quantum':
        eff_rate, out_error, out_qubits = ED_n_k_d(in_error=infidelity(in_error), n=n, k=k, d=d, printing=printing)
    
    rate_seq *= eff_rate # K/N
    total_memory_seq += K_seq*n # sum(K_{l-1}*n_{l}) - OLD
    memory_usage = max(n*K_seq, (n-1)*K_seq + prev_memory_usage) # M_l = max(n_l*K_{l-1}, (n_l-1)*K_{l-1}+M_{l-1})
    K_seq *= out_qubits # K_{l}
    

    # check if failure:
    # if infidelity(out_error) > infidelity(in_error) or total_memory_seq > max_memory:
    if infidelity(out_error) > infidelity(in_error) or memory_usage > max_memory:
        result = 'failure'

    # elif mpmath.mpf(infidelity(out_error)/K_seq) < target_output_error and total_memory_seq <= max_memory:
    elif mpmath.mpf(infidelity(out_error)/K_seq) < target_output_error:
        result = 'success'
    else:
        result = 'continue'
    return result, out_error, max_memory, target_output_error, rate_seq, total_memory_seq,  K_seq, memory_usage
            

def DFS_search(codes, max_levels, current_sequence, evaluate_function, in_error, max_memory, target_output_error, rate_seq, total_memory_seq,  K_seq, previous_code = [0,0,0,''], memory_current_sequence = [0]):
    successful_sequences = []
    
    for code in codes:
        if previous_code[3] == "Quantum" and code[3] == "Classical": # We cannot go from a quantum code to a classical code:
            continue
        code_sequence = current_sequence + [code]
        result, out_error, max_memory_new, target_output_error_new, rate_seq_new, total_memory_seq_new, K_seq_new, memory_usage = evaluate_function(code, 
                                        in_error, max_memory, target_output_error, rate_seq, total_memory_seq,  K_seq, prev_memory_usage=memory_current_sequence[-1])
        memory_sequence = memory_current_sequence + [memory_usage]
        if result == "failure":
            continue
        elif result == "continue" and len(code_sequence) < max_levels:
            sequences = DFS_search(codes, max_levels, code_sequence, evaluate_function, out_error, max_memory_new, target_output_error_new, rate_seq_new, total_memory_seq_new,  
                                K_seq_new, previous_code=code, memory_current_sequence=memory_sequence)
            successful_sequences.extend(sequences)
        
        elif result == "success":
            # create a class of the good sequence:
            sequence_result = Sequence_results(code_sequence, infidelity(out_error), total_memory_seq_new, overhead = mpmath.mpf(1/rate_seq_new), K_seq=K_seq_new, memory_sequence=memory_sequence)
            successful_sequences.append(sequence_result)
        
        
    return successful_sequences


### old functions:
def ED_C_2_1_2(p): # [2,1,2]
    pI = mpmath.mpf(p[0])
    pX = mpmath.mpf(p[1])
    pZ = mpmath.mpf(p[2])
    pY = mpmath.mpf(p[3])

    LpI = pI ** 2 + pX ** 2 # /norm = no error on the output
    LpX = 2 * pI * pX
    LpZ = pY ** 2 + pZ ** 2
    LpY = 2 * pY * pZ
    norm = LpI + LpX + LpZ + LpY
    p_reject = mpmath.mpf(1) - norm # rejection probability
    rate = (mpmath.mpf(1) / mpmath.mpf(2)) * (mpmath.mpf(1) - p_reject)
    return rate, [LpI/norm, LpX/norm, LpZ/norm, LpY/norm]

def ED_4_1_2(in_error): # out error is a vector
    L1_rate, L1_out = ED_C_2_1_2(depolarizing(in_error))
    L2_rate, L2_out = ED_C_2_1_2(hadamard(L1_out))
    out_qubits = 1
    return L2_rate * L1_rate, hadamard(L2_out), out_qubits

def ED_2n_1_n(n, in_error, basis = 'Z'): # Concat of 2 repetition codes, with H in the middle. Out error is a vector
    L1_rate, L1_out = ED_C_n_1_n(n, depolarizing(in_error))
    L2_rate, L2_out = ED_C_n_1_n(n, hadamard(L1_out))
    out_qubits = 1
    return L2_rate * L1_rate, hadamard(L2_out), out_qubits

def binomial_pmf(k, n, p): # Calculate binomial probability mass function (PMF)
    binomial_coeff = sp.binomial(n, k)
    return mpmath.mpf(binomial_coeff) * (p ** k) * ((1 - p) ** (n - k))
