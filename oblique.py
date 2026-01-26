# oblique.py
import numpy as np
import math

def sutherland_mu(T, mu_ref=1.716e-5, T_ref=273.15, S=110.4):
    """
    Sutherland's law for air (Pa·s).
    Works with scalars or numpy arrays.
    """
    T = np.asarray(T, dtype=float)
    return mu_ref * (T / T_ref)**1.5 * (T_ref + S) / (T + S)

def rasmussen(gamma, theta, M_inf):
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

def post_shock_oblique(M_inf, beta, p1, T1, p_cone, gamma=1.4, R=287.058):
    """
    Vectorized oblique-shock post-shock properties and optional
    isentropic expansion to a specified cone surface pressure p_cone.

    Parameters
    ----------
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

    Returns
    -------
    dict
        Contains arrays for the post-shock state (suffix 2) and, if `p_cone` supplied,
        the expanded cone-surface state (suffix 3). Also includes normalized viscous
        parameter 'j2' (post-shock) and 'j3' (after expansion), and Mach numbers.
    """
    beta = np.asarray(beta, dtype=float)

    # Freestream
    a1 = np.sqrt(gamma * R * T1)
    U1 = M_inf * a1
    rho1 = p1 / (R * T1)

    # Normal Mach number (array)
    Mn1 = M_inf * np.sin(beta)

    # Mask for valid attached shock (normal component > 1)
    valid = Mn1 > 1.0

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

    # Early exit if nothing valid
    if not np.any(valid):
        result = {
            'rho2': rho2,
            'u2': u2,
            'mu2': mu2,
            'p2': p2,
            'T2': T2,
            'Mn1': Mn1,
            'j2': j2,
            'M2': M2
        }
        # no expansion possible either
        return result

    # --- Normal shock relations (only where valid) ---
    Mn1v = Mn1[valid]

    # Pressure ratio across normal shock
    p2_p1 = 1.0 + 2.0 * gamma / (gamma + 1.0) * (Mn1v**2 - 1.0)
    # Density ratio across normal shock
    rho2_rho1 = ((gamma + 1.0) * Mn1v**2) / ((gamma - 1.0) * Mn1v**2 + 2.0)
    # Temperature ratio
    T2_T1 = p2_p1 / rho2_rho1

    p2[valid]   = p1 * p2_p1
    rho2[valid] = rho1 * rho2_rho1
    T2[valid]   = T1 * T2_T1

    # Velocities: decompose into normal/tangential components relative to shock
    U1_arr = U1  # scalar
    Vn1 = U1_arr * np.sin(beta[valid])
    Vt1 = U1_arr * np.cos(beta[valid])

    Vn2 = Vn1 * (rho1 / rho2[valid])    # normal velocity from continuity across shock
    Vt2 = Vt1                            # tangential velocity conserved
    u2[valid] = np.sqrt(Vn2**2 + Vt2**2)

    # Viscosity via Sutherland
    mu2[valid] = sutherland_mu(T2[valid])
    mu1 = sutherland_mu(T1)

    # Non-dimensional parameter j2 = (u2/U1) * (rho2/rho1) * (mu2/mu1)
    mue = mu2[valid]/mu1
    rhoe = rho2[valid]/rho1
    ue = u2[valid]/U1_arr

    a2[valid] = np.sqrt(gamma * R * T2[valid])
    M2[valid] = u2[valid]/a2[valid]
    j2[valid] = ue * rhoe * mue

    # Prepare result dict for post-shock
    result = {
        'rho2': rho2,
        'u2': u2,
        'mu2': mu2,
        'p2': p2,
        'T2': T2,
        'Mn1': Mn1,
        'j2': j2,
        'M2': M2
    }

    # --- Isentropic expansion to cone surface pressure p_cone ---
    # allow p_cone to be scalar or array broadcastable with beta
    p3 = np.broadcast_to(np.asarray(p_cone, dtype=float), shape)

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

    # Perform expansion only where post-shock state is valid (same valid mask)
    # and where p3 is finite/positive. We'll compute using the arrays indexed by 'valid'.
    valid_and_finite = valid & np.isfinite(p3) & (p3 > 0.0)

    if np.any(valid_and_finite):
        idx = np.where(valid_and_finite)
        # p2 and others indexed by valid mask earlier are available; but here we must index full arrays
        p2v = p2[valid_and_finite]
        rho2v = rho2[valid_and_finite]
        T2v = T2[valid_and_finite]
        u2v = u2[valid_and_finite]
        p3v = p3[valid_and_finite]

        # isentropic relation: p / rho^gamma = const -> rho3 = rho2 * (p3/p2)^(1/gamma)
        rho3v = rho2v * (p3v / p2v)**(1.0 / gamma)

        # temperature from ideal gas
        T3v = p3v / (rho3v * R)

        # stagnation (total) temperature in post-shock state: T0 = T2 + u2^2/(2*cp)
        T0v = T2v + u2v**2 / (2.0 * cp)

        # compute final velocity from energy conservation: u3 = sqrt(2*cp*(T0 - T3))
        # ensure non-negative argument for sqrt; if negative, set to zero (physical limit reached)
        deltaT = T0v - T3v
        # numerical tolerance: treat tiny negatives as zero
        deltaT_clamped = np.where(deltaT > 0.0, deltaT, 0.0)
        u3v = np.sqrt(2.0 * cp * deltaT_clamped)

        # local sound speed and Mach number after expansion
        a3v = np.sqrt(gamma * R * T3v)
        M3v = np.where(a3v > 0.0, u3v / a3v, np.nan)

        mu3v = sutherland_mu(T3v)

        # j3 normalized viscous parameter analogous to j2
        j3v = (u3v / U1_arr) * (rho3v / rho1) * (mu3v / mu1)

        # fill back into full arrays
        rho3[valid_and_finite] = rho3v
        T3[valid_and_finite] = T3v
        u3[valid_and_finite] = u3v
        mu3[valid_and_finite] = mu3v
        j3[valid_and_finite] = j3v
        a3[valid_and_finite] = a3v
        M3[valid_and_finite] = M3v

    # add expansion results to result dict
    result.update({
        'rho3': rho3,
        'u3': u3,
        'mu3': mu3,
        'p3': p3,
        'T3': T3,
        'j3': j3,
        'M3': M3,
        'a3': a3
    })

    return j3, M3 

