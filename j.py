# oblique.py
import numpy as np
import math
from shape import *

def sutherland_mu(T, mu_ref=1.716e-5, T_ref=273.15, S=110.4):
    """
    sutherland's law for air (Pa·s).
    note that in order to replicate Rotta's result a linear viscosity
    model is required. use of sutherland's will cause higher temperature
    conditions (roughly above M12) to diverge from rotta, probably in a 
    good way though. to replicate rotta, change the return line
    """
    T = np.asarray(T, dtype=float)
    return mu_ref * (T / T_ref)**1.5 * (T_ref + S) / (T + S)
    #return mu_ref * (T / T_ref)

def rasmussen(gamma, theta, M_inf):
    """
    rasmussens Cp correlation for cones. used to determine the cone surface
    pressure for the isentropic expansion
    """
    # expect theta in rad
    K = M_inf*theta 

    cp  = gamma + 1
    cp /= 2
    cp += 1/(K**2)
    cp  = math.log(cp)
    cp *= (gamma+1)*(K**2) + 2
    cp /= (gamma-1)*(K**2) + 2
    cp += 1                     
    cp *= theta**2
    
    return cp

def modnewton(p02, p_inf, rho_inf, V_inf, theta):
    """
    modified newton Cp distribution. currently unused, but probably an 
    improvement over rasmussen
    """
    cp_max = p02 - p_inf
    cp_max /= 0.5*rho_inf*(V_inf)**2
    a = cp_max[0]  
    #a = cp_max
    cp = a*(np.sin(theta)**2)
    return cp
    

def edge(M_inf, beta, p1, T1, y_bar, theta, p_cone,gamma=1.4, R=287.058):
    """
    Vectorized oblique-shock post-shock properties and optional
    isentropic expansion to a specified cone surface pressure p_cone.

    M_inf : float
        Freestream Mach number.
    beta : float or ndarray
        Shock angle(s) in radians.
    p1 : float
        Freestream static pressure (Pa).
    T1 : float
        Freestream static temperature (K).
    p_cone : None, float, or ndarray, optional
        Cone surface pressure(s) to expand to (Pa). If None, no expansion is performed.
        If array, must be broadcastable to `beta`.
    gamma : float
        Ratio of specific heats.
    R : float
        Gas constant (J/kg/K).
    """
    beta = np.asarray(beta, dtype=float)

    # Freestream
    a1 = np.sqrt(gamma * R * T1)
    U1 = M_inf * a1
    rho1 = p1 / (R * T1)

    # Normal Mach number (array)
    Mn1 = M_inf * np.sin(beta)

    # Allocate output arrays (same shape as beta)
    shape = np.shape(beta)
    rho2 = np.full(shape, np.nan, dtype=float)
    u2   = np.full(shape, np.nan, dtype=float)
    mu2  = np.full(shape, np.nan, dtype=float)
    p2   = np.full(shape, np.nan, dtype=float)
    T2   = np.full(shape, np.nan, dtype=float)
    j2   = np.full(shape, np.nan, dtype=float)
    a2   = np.full(shape, np.nan, dtype=float)
    M2   = np.full(shape, np.nan, dtype=float)

    # --- Normal shock relations        
    Mn1v = Mn1           

    # Pressure ratio across normal shock
    p2_p1 = 1.0 + 2.0 * gamma / (gamma + 1.0) * (Mn1v**2 - 1.0)
    # Density ratio across normal shock
    rho2_rho1 = ((gamma + 1.0) * Mn1v**2) / ((gamma - 1.0) * Mn1v**2 + 2.0)
    # Temperature ratio
    T2_T1 = p2_p1 / rho2_rho1

    p2   = p1 * p2_p1
    rho2 = rho1 * rho2_rho1
    T2   = T1 * T2_T1

    # Velocities: decompose into normal/tangential components relative to shock
    U1_arr = U1  # scalar
    Vn1 = U1_arr * np.sin(beta)
    Vt1 = U1_arr * np.cos(beta)

    Vn2 = Vn1 * (rho1 / rho2)    # normal velocity from continuity across shock
    Vt2 = Vt1                    # tangential velocity conserved
    u2 = np.sqrt(Vn2**2 + Vt2**2)

    # Viscosity via Sutherland
    mu2 = sutherland_mu(T2)
    mu1 = sutherland_mu(T1)

    # Non-dimensional parameter j2 = (u2/U1) * (rho2/rho1) * (mu2/mu1)
    mue = mu2/mu1
    rhoe = rho2/rho1
    ue = u2/U1_arr

    a2 = np.sqrt(gamma * R * T2)
    M2 = u2/a2
    j2 = ue * rhoe * mue

    # determine the cone surface pressure

    p02 = p2 * (1.0 + 0.5*(gamma - 1.0)*M2**2)**(gamma/(gamma - 1.0)) 
    cp_surface = rasmussen(gamma, theta, M_inf)    
    q = 0.5*rho1*(U1**2)
    p3 = q*cp_surface + p1
    
    # --- Isentropic expansion to cone surface pressure p_cone ---
    # allow p_cone to be scalar or array broadcastable with beta

    #p3 = np.broadcast_to(np.asarray(p_cone, dtype=float), shape)

    # allocate outputs for expanded state
    rho3 = np.full(shape, np.nan, dtype=float)
    T3   = np.full(shape, np.nan, dtype=float)
    u3   = np.full(shape, np.nan, dtype=float)
    mu3  = np.full(shape, np.nan, dtype=float)
    j3   = np.full(shape, np.nan, dtype=float)
    a3   = np.full(shape, np.nan, dtype=float)
    M3   = np.full(shape, np.nan, dtype=float)

    # Specific heat at constant pressure
    cp = gamma * R / (gamma - 1.0)

    # p2 and others indexed by valid mask earlier are available; but here we must index full arrays
    p2v   = p2
    rho2v = rho2
    T2v   = T2
    u2v   = u2
    p3v   = p3
    
    # isentropic relation: p / rho^gamma = const -> rho3 = rho2 * (p3/p2)^(1/gamma)
    rho3v = rho2v * (p3v / p2v)**(1.0 / gamma)

    # temperature from ideal gas
    T3v = p3v / (rho3v * R)

    # stagnation (total) temperature in post-shock state: T0 = T2 + u2^2/(2*cp)
    T0v = T2v + u2v**2 / (2.0 * cp)

    # compute final velocity from energy conservation: u3 = sqrt(2*cp*(T0 - T3))
    # ensure non-negative argument for sqrt; if negative, set to zero (physical limit reached)
    deltaT = T0v - T3v

    u3v = np.sqrt(2.0 * cp * deltaT)

    # local sound speed and Mach number after expansion
    a3v = np.sqrt(gamma * R * T3v)
    M3v = u3v / a3v

    mu3v = sutherland_mu(T3v)

    # j3 is j(y_bar) as used by rotta in the paper   
    j3v = (u3v / U1_arr) * (rho3v / rho1) * (mu3v / mu1)
    
    # fill back into full arrays
    rho3 = rho3v
    T3   = T3v
    u3   = u3v
    mu3  = mu3v
    j3   = j3v
    a3   = a3v
    M3   = M3v

    return j3, M3 

