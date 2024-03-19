from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems import get_problem
from pymoo.optimize import minimize
import numpy as np
from pymoo.termination import get_termination
from pymoo.core.callback import Callback


class cb(Callback): # Use this callback while running minimize() on result object
    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []

    def notify(self, algorithm):
        self.data["best"].append(algorithm.pop.get("F"))



def check_if_conv(k , eps ,  cb): # returns min number of iterations required for convergence
    f = [np.mean(cb.data["best"][i]) for i in range(len(cb.data["best"]))]

    for i in range(len(f) - k):
        # ackley
        if ((max(f[i:i + k]) - min(f[i:i+k])) < eps):
            return i+k
    return -1

if __name__== "__main__":
    problem = get_problem("ackley", n_var=2, a=20, b=1/5, c=2 * np.pi)

    algorithm = GA(
        pop_size=100,
        eliminate_duplicates=True)

    tc = get_termination("n_gen", 300)



    res = minimize(problem,
                algorithm,
                tc,
                seed=1,
                callback=cb(),
                verbose=True
                )

    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

    print("Convergence in " + str(check_if_conv(5  , 1e-5, res.algorithm.callback)))

