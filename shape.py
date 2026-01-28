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
    
    #return np.arctan(0.992*(c_dt**0.5)*((np.cos(theta_b)/y_bar)**1.174)), x
    return beta,x


# Determine the conical shock angle (page 5)

def conical_shock_angle(tb,M_inf):
    # expect tb input in radians
    del_c  = M_inf*np.sin(tb) - 3.43
    del_c *= 1.01                    
    del_c += 4.0                     
    del_c /= M_inf                   
    del_c  = np.arcsin(del_c) #in radians

    return del_c

def surface(y_bar, theta):
    dum = y_bar.copy()
    dum[dum>1]=1
    angle = np.arctan(dum)
    angle = 90 - np.rad2deg(angle)
    angle = np.deg2rad(angle)
    angle[angle < theta] = theta

    return angle 
