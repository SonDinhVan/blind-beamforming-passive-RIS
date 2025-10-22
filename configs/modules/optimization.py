"""
This class provides codes for several beamforming algorithms, including:
- Perfect CSI
- Random Max Sampling algorithm (RMS)
- RFocus
- Conditional Sample Mean (CSM)
- Grouped CSM (GCSM) is NOT implemented here due to many stages included in main protocols.
  This is a drawback of this algorithm.
  It can be found in module system, and is included in our benchmarks.
- Our optimization stage
"""
import numpy as np
from scipy.linalg import pinv, eigh

def perfect_csi(h_0: np.array, h_: np.array) -> np.array:
    """
    Solve the phase shifts when perfect CSI is available.
    Maximize |h_0 + h_ x |^2
    Subject to x in {-1, 1}^N

    Args:
        h_0 (np.array): the background channel
        h_ (np.array): cascaded channel via RIS
            h_0 and h_ must be column vectors.

    Returns:
        np.array: x and theta
    """
    N = h_.shape[0]  # Number of reflective elements
    h = np.zeros(N + 1, dtype=complex)
    h[0] = h_0
    h[1:] = h_.flatten()  # Flatten h_vec to ensure 1D array compatibility
    phi = np.angle(h)  # Phases in radians, [-π, π]
    events = []
    for n in range(N + 1):
        psi_minus = (phi[n] - np.pi / 2) % (2 * np.pi)
        psi_plus = (phi[n] + np.pi / 2) % (2 * np.pi)
        events.append((psi_minus, n, 'minus'))
        events.append((psi_plus, n, 'plus'))

    events.sort(key=lambda x: x[0])
    v = np.stack((np.real(h), np.imag(h)), axis=1)  # Shape: (N+1, 2)
    s = np.sign(np.real(h))
    s[s == 0] = np.sign(np.imag(h[s == 0]))
    w = np.sum(v[s > 0], axis=0) - np.sum(v[s < 0], axis=0)
    candidates = []
    theta_prev = events[-1][0] - 2 * np.pi if len(events) > 0 else -2 * np.pi
    for m in range(len(events)):
        theta_m = events[m][0]
        n_m = events[m][1]
        type_m = events[m][2]
        
        psi_m = np.arctan2(w[1], w[0]) % (2 * np.pi)
        norm_w = np.linalg.norm(w)
        
        if theta_prev < 0 and theta_m >= 0:
            in_arc = (psi_m > theta_prev + 2 * np.pi) or (psi_m < theta_m)
        else:
            in_arc = theta_prev < psi_m < theta_m
        
        if in_arc and norm_w > 0:
            y_m_star = np.array([np.cos(psi_m), np.sin(psi_m)])
            g_m = norm_w
        else:
            y_start = np.array([np.cos(theta_prev), np.sin(theta_prev)])
            y_end = np.array([np.cos(theta_m), np.sin(theta_m)])
            g_start = np.dot(w, y_start)
            g_end = np.dot(w, y_end)
            if g_start > g_end:
                y_m_star = y_start
                g_m = g_start
            else:
                y_m_star = y_end
                g_m = g_end
        
        candidates.append((y_m_star, g_m))
        
        if type_m == 'minus':
            s[n_m] = 1
            w += 2 * v[n_m]
        else:
            s[n_m] = -1
            w -= 2 * v[n_m]
        
        theta_prev = theta_m
    y_star, g_max = max(candidates, key=lambda x: x[1])

    v0_y = np.dot(v[0], y_star)
    vn_y = np.dot(v[1:], y_star)
    x = np.sign(v0_y * vn_y) if v0_y !=0 else np.sign(vn_y)
    theta = np.arccos(x)
    
    return theta, x

def RMS(X: np.array, y: np.array) -> np.array:
    """
    Random max sampling algorithm

    Args:
        X, y: feature and label data.
              Shape of X = (size of RIS, number of samples)

    Returns:
        np.array: the best vector x.
    """
    return X[:, np.argmax(y)].reshape(-1, 1)

def RFocus(X: np.array, y: np.array) -> np.array:
    """
    RFocus algorithm.
    Arun, V. and Balakrishnan, H., 2020. {RFocus}: Beamforming using thousands of passive antennas.
    In 17th USENIX symposium on networked systems design and implementation (NSDI 20) (pp. 1047-1061).

    Args:
        X, y: feature and label data.
              Shape of X = (size of RIS, number of samples)

    Returns:
        np.array: the best vector x.
    """
    N = X.shape[0]
    final_state = np.zeros(shape=(N, 1))
    m = np.mean(y)
    for i in range(N):
        vote_on, vote_off = 0, 0
        for j in range(X.shape[1]):
            if (X[i, j] == 1 and y[j] > m) or (X[i, j] == -1 and y[j] < m):
                vote_on += 1
            else:
                vote_off += 1
        final_state[i] = 1 if vote_on > vote_off else -1
    
    return final_state

def CSM(X: np.array, y: np.array) -> np.array:
    """
    Conditional Sample Mean algorithm.
    S. Ren, K. Shen, Y. Zhang, X. Li, X. Chen, and Z.-Q. Luo, “Configuring intelligent reflecting surface
    with performance guarantees: Blind beamforming,” IEEE Trans. Wireless Commun., vol. 22, no. 5, pp. 3355–3370, 2023.

    Args:
        X, y: feature and label data.
              Shape of X = (size of RIS, number of samples)

    Returns:
        np.array: the best vector x.
    """
    N = X.shape[0]
    N_samples = X.shape[1]
    final_state = np.zeros(shape=(N, 1))
    for i in range(N):
        mean_of_1 = np.sum((X[i] == 1) * y.reshape(N_samples)) / np.sum(X[i] == 1)
        mean_of_minus_1 = np.sum((X[i] == -1) * y.reshape(N_samples)) / np.sum(X[i] == -1)
        final_state[i] = 1 if mean_of_1 > mean_of_minus_1 else -1
    
    return final_state

def compute_V_and_s(M: np.array, w: np.array):
    """
    Compute V and s for the quadratic form x^T M x + x^T w,
    where M is positive semidefinite, using eigenvalue decomposition for efficiency and truncation.
    
    Parameters:
        M (np.ndarray): Positive semidefinite matrix (N x N).
        w (np.ndarray): Vector (N x 1 or 1D array).
    
    Returns:
        V (np.ndarray): Matrix such that V.T @ V ≈ M (r x N, with r <= N).
        s (np.ndarray): Vector s = (1/2) (V.T)^+ w (r x 1).
    """
    # Ensure w is column vector
    w = w.flatten()[:, np.newaxis] if w.ndim == 1 else w
    
    # Eigen decomposition
    lamb, U = eigh(M)
    lamb_pos = lamb[-2:]
    U_pos = U[:, -2:]

    sqrt_lamb_pos = np.sqrt(lamb_pos)
    V = np.diag(sqrt_lamb_pos) @ U_pos.T  # r x N
    
    # Pseudoinverse of V.T
    VT_pinv = pinv(V.T)
    
    s = (1 / 2) * VT_pinv @ w
    
    return V, s.flatten()  # s as 1D array

def solve_max_norm_squared_2d(V: np.array, s: np.array):
    """
    Exactly maximize ||V x + s||^2 over x in {-1,1}^N, assuming V is 2 x N and s is 2,.
    
    Parameters:
        V (np.ndarray): 2 x N matrix.
        s (np.ndarray): 2-element array.
    
    Returns:
        x_opt (np.ndarray): Optimal x (N,).
        opt_norm_squared (float): Optimal ||V x_opt + s||^2.
    """
    assert V.shape[0] == 2 and len(s) == 2, "Assumes 2D."
    N = V.shape[1]
    v = [s.copy()]  # v0 = s
    v += [V[:, n].copy() for n in range(N)]  # v1 to vN = columns of V
    
    # Compute boundary points
    boundaries = []
    for n in range(N + 1):
        phi = np.arctan2(v[n][1], v[n][0])
        ang_minus = (phi - np.pi / 2) % (2 * np.pi)
        ang_plus = (phi + np.pi / 2) % (2 * np.pi)
        boundaries.append((ang_minus, n, -1))  # type -1 for y^-
        boundaries.append((ang_plus, n, 1))     # type 1 for y^+
    
    boundaries.sort(key=lambda x: x[0])  # Sort by angle
    ang = [b[0] for b in boundaries]
    
    # Initialize w for the wrapping arc (midpoint for signs)
    start_ang_last = ang[-1]
    end_ang_first = ang[0] + 2 * np.pi
    mid_ang = (start_ang_last + end_ang_first) / 2 % (2 * np.pi)
    y_mid = np.array([np.cos(mid_ang), np.sin(mid_ang)])
    
    signs = [np.sign(np.dot(vv, y_mid)) if np.dot(vv, y_mid) != 0 else 1.0 for vv in v]
    w_current = np.sum([signs[i] * v[i] for i in range(N + 1)], axis=0)
    
    # Collect w for each arc, start/end angles
    w_all = [w_current.copy()]
    start_angs = [start_ang_last % (2 * np.pi)]
    end_angs = [ang[0]]
    
    for i in range(len(boundaries)):
        _, n, sign_type = boundaries[i]
        if sign_type == 1:  # crossing y^+
            w_current -= 2 * v[n]
        else:  # crossing y^-
            w_current += 2 * v[n]
        w_all.append(w_current.copy())
        
        next_i = (i + 1) % len(ang)
        start_angs.append(ang[i])
        end_angs.append(ang[next_i])
    
    w_all.pop()  # Last is duplicate of first, but we handle wrap separately
    
    # Find max g over arcs
    g_max = 0.0
    y_star = None
    for i in range(len(w_all)):
        w = w_all[i]
        norm_w = np.linalg.norm(w)
        start = start_angs[i]
        end = end_angs[i]
        if i == 0:  # Wrapping arc
            end += 2 * np.pi
        
        y_start = np.array([np.cos(start), np.sin(start)])
        y_end = np.array([np.cos(end % (2 * np.pi)), np.sin(end % (2 * np.pi))])
        
        g_start = np.dot(w, y_start)
        g_end = np.dot(w, y_end)
        g_arc = max(g_start, g_end)
        y_candidate = y_end if g_end > g_start else y_start
        
        if norm_w > 1e-10:
            y_opt = w / norm_w
            ang_opt = np.arctan2(y_opt[1], y_opt[0]) % (2 * np.pi)
            if ang_opt < start:
                ang_opt += 2 * np.pi
            if start <= ang_opt < end:
                g_arc = norm_w
                y_candidate = y_opt
        
        if g_arc > g_max:
            g_max = g_arc
            y_star = y_candidate
    
    # Recover x*
    sign0 = np.sign(np.dot(v[0], y_star))
    if sign0 == 0:
        sign0 = 1.0
    x_opt = np.array([sign0 * (np.sign(np.dot(vv, y_star)) if np.dot(vv, y_star) != 0 else 1.0) for vv in v[1:]])

    return x_opt.reshape([-1, 1])

def solve_our_optimization(M_true, w_true):
    """
    Solve max x^T M_true x + x^T w_true over x in {-1,1}^N using the 2D sweep method.
    Assumes the effective rank of M_true is 2 (as in IRS applications).
    
    Parameters:
        M_true (np.ndarray): N x N PSD matrix.
        w_true (np.ndarray): N x 1 vector.
    
    Returns:
        x_opt (np.ndarray): Optimal x.
        opt_value (float): Optimal objective value.
    """
    V, s = compute_V_and_s(M_true, w_true)
    
    # Assume/check effective dimension is 2
    r = V.shape[0]
    if r != 2:
        raise ValueError(f"Effective dimension is {r}, but the sweep method assumes 2D. Use a heuristic for higher dimensions.")
    
    return solve_max_norm_squared_2d(V, s)
