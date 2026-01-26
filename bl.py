import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

def self_similar(beta,g_w):
    
    # g_w = 0 denotes an adiabatic wall, this could be changed
    # if desired (see Anderson HHTGD page 284 for details)

    #g_w = 0.0

    # the maximum value of eta (self similar parameter) that 
    # we will allow the solver to integrate to. just needs to
    # be large enough that we reach steady state

    eta_max = 50.0

    # solver tolerance 

    eps_f = eps_g = 1e-6

    # boundary layer odes (simplified form of eqns [9] and [10]
    # from Lees paper). we need to express these equations as a 
    # first order system (i.e. only contain first derivative terms)
    # so we can integrate forward through eta. I have chosen to define 
    # the vector y as below, so we can integrate forward like:
    #       y_(i+1) = y_(i) + yp_(i)*deta
    # the odes function simply returns yp for a given y

    def odes(eta, y):
        f, fp, fpp, g, gp = y
        return [
            fp,
            fpp,
            -f*fpp - beta*(g - fp**2),
            gp,
            -f*gp
        ]
    
    # edge event for the shooting method
    # stop when fp = 1 and g = 1 within tolerance of eps 

    def edge_event(eta, y):
        return max(abs(y[1]-1)-eps_f, abs(y[3]-1)-eps_g)
    edge_event.terminal = True
    edge_event.direction = -1
    
    # solve the boundary layer equations using a shooting method
    # see Anderson HHTGD (2nd edition) page 289. we want to find 
    # fpp0 and gp0 that satisfy our choice of boundary conditions

    # the residuals function returns the difference (residual) between
    # the fp(eta_max) and g(eta_max) values and 1 (we require them to be
    # 1 as these are the boundary conditions). we then pass this residuals
    # function to scipy and let it solve it for us

    def residuals(ab):
        fpp0, gp0 = ab
        y0 = [0, 0, fpp0, g_w, gp0]
    
        sol = solve_ivp(odes, (0, eta_max), y0,
                        method='BDF', events=edge_event)
    
        y_edge = sol.y_events[0][0] if sol.t_events[0].size else sol.y[:, -1]
        return [y_edge[1]-1, y_edge[3]-1]

    # optimize residuals function with initial guesses for fpp0 and gp0 as 
    # specified
    
    sol = root(residuals, [0.33, (1-g_w)/10])
    
    fpp0, gp0 = sol.x
    
    # now that the odes are fully defined we can solve to find f(eta_e)

    y0 = [0, 0, fpp0, g_w, gp0]
    sol = solve_ivp(odes, (0, eta_max), y0,
                    method='BDF', events=edge_event)
    
    y_edge = sol.y_events[0][0]
    eta_edge = sol.t_events[0][0]
    
    # output solution to text
    np.savetxt(
        "bl.txt",
        np.column_stack((sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3], sol.y[4])),
        header="eta f fp fpp g gp",
        comments=""
        )
    return y_edge[0]

