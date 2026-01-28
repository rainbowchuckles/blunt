import numpy as np
from scipy.integrate import solve_ivp, cumtrapz
from scipy.optimize import root, brentq
from j import *                   
from f import *                          
from shape import *   

# problem definition  

M_inf    = 15.0          # freestream Mach bnumber
gamma    = 1.4           # ratio of specific heats
tb       = 20            # cone half angle (degrees)
p1       = 137.9         # Pa
T1       = 266.15        # K
b        = 0.90          # self similar boundary layer parameter
R_gas    = 287           # Gas constant (assume air)
r0       = 0.15          # Nose radius (m)

# shouldn't be any need to edit below this line
# ---------------------------------------------

# Other freestream properties that we might need

rho1   = p1/(R_gas*T1)
a1     = (gamma*R_gas*T1)**0.5
V1     = M_inf*a1
Re_inf = rho1*V1*r0/(sutherland_mu(T1)) 

# Determine the conical shock angle (page 5)

tb = np.deg2rad(tb) # convert cone half angle to radian 

del_c = conical_shock_angle(tb,M_inf)

# Now determine the sensible upper integration limit ybar_c
# This is the non-dimensional swallowing height

cdt = 2.0 - np.cos(tb)**2

ybar_c = np.sin(del_c) 
ybar_c = ybar_c**2          
ybar_c = 1/ybar_c          
ybar_c -= 1
ybar_c *= 0.984*cdt
ybar_c = ybar_c**0.426
ybar_c *= np.cos(tb)

# Evaluate the shock angle as a function of y_bar

y_bar = np.linspace(1e-6, ybar_c, 1000) 
beta, x_bar = shock_angle_from_y(y_bar, tb, cdt)

# Compute j(y_bar) by determining the post-shock conditions as a function of y_bar
# and then expanding these to the inviscid cone surface pressure                

j, Me = edge(M_inf, beta, p1, T1, y_bar,tb,p_cone)

# Get f(eta_e) from the self similar model 

f = self_similar(b,0.0)

# Form integrand

integrand = (y_bar**3) / j

# Numerical integration

I = cumtrapz(integrand, y_bar, initial=0)

# Now we have everything we need to evaluate S_bar/Re_inf

sr  = f**2
sr *= np.sin(tb)**2
sr  = 1/sr
sr *= 1.5
sr *= I
sr  = sr**(1/3)

# Output

np.savetxt(
    "output.dat",
    np.column_stack((sr, np.rad2deg(beta),Me)),
    header="sr b Me",
    comments=""
)

