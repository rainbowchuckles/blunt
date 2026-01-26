import numpy as np

# Klaimon (ref 8) shock shape correlation 

def klaimon_xbar(theta_b, c_dt, y_bar):
    # theta_b in radians
    x_bar  = y_bar/(1.424*np.cos(theta_b))
    x_bar  = x_bar**(1.0/0.46)           
    x_bar *= np.cos(theta_b)/np.sqrt(c_dt)
    return x_bar                                                

# Get shock angle from the correlation as a function of ybar

def shock_angle_from_y(y_bar, theta_b, c_dt):
    y = np.asarray(y_bar)
    x = klaimon_xbar(theta_b, c_dt, y)                         
    

    dx_dy = np.gradient(x, y)
    dydx = 1.0 / dx_dy
    beta = np.arctan(dydx)
    np.savetxt(
    "shock_shape.dat",
    np.column_stack((x, y, np.rad2deg(beta))),
    header="x y beta",
    comments="")
    return np.abs(beta)


# Determine the conical shock angle (page 5)

def conical_shock_angle(tb,M_inf):
    # expect tb input in radians
    del_c  = M_inf*np.sin(tb) - 3.43
    del_c *= 1.01                    
    del_c += 4.0                     
    del_c /= M_inf                   
    del_c  = np.arcsin(del_c) #in radians

    return del_c
