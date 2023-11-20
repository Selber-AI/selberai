import scipy
import random



def solve(opt_prob: dict[callable, list[callable], list[callable]]) -> dict:
  """
  Solves an optimization problem.
  """
  
  
  objective = opt_prob['obj_fun']
  x0 = [1, 2, 3, 4]
  
  constraints = []
  for func in opt_prob.eq_func_list:
    constraints.append(
      {'type': 'eq', 'fun': func}
    )
  for func in opt_prob.ineq_func_list:
    constraints.append(
      {'type': 'ineq', 'fun': func}
    )
  
  sol = scipy.optimize.minimize(objective, x0, constraints=constraints)
  solution = {
    'value': sol.fun,
    'var': sol.x
  }
  
  
  return solution 
