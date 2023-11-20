import scipy
import random



def solve(opt_prob:dict[callable, list[callable], list[callable]]) -> dict:
  """
  Solves an optimization problem.
  """
  
  
  ###
  # SOLVE problem here.
  ### 
  
  ### Note: scipy processes problems NOT in standard format. It has reversed
  # signs for the inequality constrains compared to standard format.
  objective = opt_prob['obj_func']
  constraints = []
  for func in opt_prob['eq_func_list']:
    constraints.append(
      {'type': 'eq', 'fun': func}
    )
  for func in opt_prob['ineq_func_list']:
    constraints.append(
      {'type': 'ineq', 'fun': func}
    )
  
  x0 = [1, 2, 3, 4]
  
  sol = scipy.optimize.minimize(objective, x0, constraints=constraints)
  solution = {
    'value': sol.fun,
    'variables': sol.x
  }
  
  
  return solution 
