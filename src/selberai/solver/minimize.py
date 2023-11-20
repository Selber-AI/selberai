import dc3_plus
import numpy as np



def minimize(obj_func:callable, eq_func_list:list[callable]=None, 
  ineq_func_list:list[callable]=None, method='dc3_plus') -> dict:
  
  """
  Function that solves an optimization problem passed in standard format.
  
  min f(x)                (obj_func)
  x
  
  s.t.  g(x) = 0          (eq_func_list)
        h(x) <= 0         (ineq_func_list)
  
  
  Methods can be chosen from the following list:
    - 'dc3_plus'
    
  Returns a dictionary containing the objective function and variables
  """
  
  opt_prob = {
    'obj_func' : obj_func,
    'eq_func_list' : eq_func_list,
    'ineq_func_list' : ineq_func_list
  }
  
  
  if method == 'dc3_plus':
    solution = dc3_plus.solve(opt_prob)
  
  
  
  return solution
  
  
