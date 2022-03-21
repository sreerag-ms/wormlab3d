import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.linalg import norm
from scipy.integrate import cumulative_trapezoid

X_orig = np.array([[0.1048975, 0., 0.],
                   [0.10545421, 0.00734377, 0.00268555],
                   [0.10601091, 0.01468778, 0.00537109],
                   [0.10656881, 0.02203238, 0.00830078],
                   [0.10712862, 0.02937829, 0.01098633],
                   [0.10769081, 0.03672659, 0.01391602],
                   [0.10825706, 0.04407859, 0.01660156],
                   [0.10882807, 0.05143631, 0.01928711],
                   [0.10940719, 0.05880165, 0.02197266],
                   [0.10999942, 0.06617618, 0.0246582],
                   [0.11060381, 0.0735631, 0.02734375],
                   [0.1112206, 0.08096755, 0.0300293],
                   [0.11185098, 0.08839428, 0.0324707],
                   [0.1124959, 0.09584737, 0.03491211],
                   [0.11315584, 0.10332823, 0.03735352],
                   [0.11383009, 0.11083567, 0.03955078],
                   [0.11451602, 0.11836672, 0.04174805],
                   [0.11520648, 0.12591732, 0.04394531],
                   [0.11588955, 0.13348353, 0.04589844],
                   [0.11654902, 0.14106262, 0.0480957],
                   [0.11716366, 0.14865267, 0.05004883],
                   [0.11771393, 0.15625083, 0.05200195],
                   [0.11818242, 0.16385365, 0.05395508],
                   [0.11855364, 0.17145705, 0.05615234],
                   [0.11881208, 0.17905581, 0.05810547],
                   [0.11894321, 0.1866442, 0.06030273],
                   [0.1189332, 0.19421577, 0.06225586],
                   [0.11876845, 0.20176315, 0.06469727],
                   [0.11843657, 0.20927715, 0.06689453],
                   [0.11792636, 0.21674657, 0.06933594],
                   [0.11722922, 0.22415924, 0.07202148],
                   [0.11633945, 0.23150074, 0.07470703],
                   [0.11524701, 0.23875487, 0.07739258],
                   [0.11394095, 0.24590266, 0.08056641],
                   [0.11241007, 0.2529223, 0.08374023],
                   [0.11064529, 0.25978887, 0.0871582],
                   [0.10864162, 0.26647675, 0.09082031],
                   [0.10640025, 0.27295947, 0.09472656],
                   [0.10392523, 0.27920842, 0.09887695],
                   [0.10122538, 0.28519785, 0.10302734],
                   [0.09831786, 0.29090607, 0.10766602],
                   [0.09522557, 0.29631722, 0.11254883],
                   [0.09197426, 0.30141914, 0.11743164],
                   [0.08858943, 0.30620289, 0.12280273],
                   [0.08509994, 0.3106643, 0.12817383],
                   [0.08153486, 0.31480432, 0.1340332],
                   [0.07792354, 0.31862915, 0.13989258],
                   [0.07429314, 0.32214987, 0.14575195],
                   [0.07067037, 0.32537985, 0.15209961],
                   [0.06708097, 0.32833552, 0.15844727],
                   [0.06354618, 0.33103895, 0.16479492],
                   [0.0600872, 0.33351469, 0.17163086],
                   [0.05672646, 0.33578777, 0.17822266],
                   [0.05348802, 0.33788574, 0.18505859],
                   [0.05039167, 0.33983648, 0.19213867],
                   [0.04745841, 0.34167218, 0.19921875],
                   [0.04470587, 0.34342706, 0.20629883],
                   [0.04214382, 0.34513366, 0.21362305],
                   [0.03977728, 0.34682155, 0.22094727],
                   [0.03760886, 0.34851408, 0.22827148],
                   [0.03563428, 0.35023463, 0.2355957],
                   [0.03384471, 0.35200536, 0.24316406],
                   [0.03222919, 0.35384643, 0.25073242],
                   [0.03076935, 0.35577869, 0.25805664],
                   [0.02944469, 0.35782111, 0.265625],
                   [0.02823162, 0.35999048, 0.27319336],
                   [0.02710891, 0.36230481, 0.28051758],
                   [0.02605939, 0.3647809, 0.2878418],
                   [0.02506471, 0.36743677, 0.29541016],
                   [0.02410603, 0.37029123, 0.30249023],
                   [0.02316546, 0.37336338, 0.30981445],
                   [0.02222991, 0.37667298, 0.31689453],
                   [0.02129126, 0.38024092, 0.32373047],
                   [0.02034974, 0.38409662, 0.33056641],
                   [0.01940632, 0.38827252, 0.3371582],
                   [0.01846409, 0.39280188, 0.34350586],
                   [0.01752996, 0.39771879, 0.34960938],
                   [0.01661181, 0.40304995, 0.35546875],
                   [0.0157218, 0.40881145, 0.3605957],
                   [0.01487541, 0.4150064, 0.36547852],
                   [0.01409483, 0.42162251, 0.36962891],
                   [0.01340747, 0.42862785, 0.37329102],
                   [0.01284504, 0.43596435, 0.37597656],
                   [0.01243496, 0.44355476, 0.37792969],
                   [0.01219797, 0.45131552, 0.37939453],
                   [0.01214457, 0.459167, 0.37988281],
                   [0.01227522, 0.46703947, 0.37988281],
                   [0.0125823, 0.47487617, 0.37915039],
                   [0.01305079, 0.48263192, 0.37792969],
                   [0.01365972, 0.49027109, 0.37597656],
                   [0.01438427, 0.49776542, 0.3737793],
                   [0.01519465, 0.50509322, 0.37109375],
                   [0.01605844, 0.51223743, 0.36791992],
                   [0.01694655, 0.51918435, 0.36425781],
                   [0.01783299, 0.5259248, 0.36035156],
                   [0.01869488, 0.53245068, 0.35595703],
                   [0.01951456, 0.53875434, 0.35131836],
                   [0.02027845, 0.54482877, 0.34643555],
                   [0.02097464, 0.55066979, 0.34106445],
                   [0.02159381, 0.55627739, 0.33569336],
                   [0.02212501, 0.56165802, 0.32983398],
                   [0.02255893, 0.56682289, 0.32397461],
                   [0.02288604, 0.57178795, 0.31787109],
                   [0.02309656, 0.57658064, 0.31152344],
                   [0.02318335, 0.58123362, 0.30517578],
                   [0.02314019, 0.58578503, 0.29882812],
                   [0.02296376, 0.59027565, 0.29223633],
                   [0.02265453, 0.59474814, 0.28588867],
                   [0.02222061, 0.59923613, 0.27954102],
                   [0.0216713, 0.60376894, 0.27294922],
                   [0.0210197, 0.60837519, 0.26660156],
                   [0.02027941, 0.61308086, 0.26049805],
                   [0.01945949, 0.61790955, 0.25415039],
                   [0.0185647, 0.62287867, 0.24829102],
                   [0.01759744, 0.62799585, 0.2421875],
                   [0.01655841, 0.63326204, 0.23657227],
                   [0.01544738, 0.63867295, 0.23095703],
                   [0.01426411, 0.64421952, 0.2253418],
                   [0.01301098, 0.64988983, 0.22021484],
                   [0.01169133, 0.65566814, 0.21484375],
                   [0.0103128, 0.66153872, 0.20996094],
                   [0.00888777, 0.66748583, 0.20483398],
                   [0.0074327, 0.67349446, 0.19995117],
                   [0.00595832, 0.67955101, 0.1953125],
                   [0.00447392, 0.68564212, 0.19042969],
                   [0.00298429, 0.69175446, 0.18579102],
                   [0.0014925, 0.697878, 0.1809082],
                   [0., 0.70400512, 0.17626953]])

X_orig = np.array([[-0.0274723, -0.00563511, -0.12673669],
                   [-0.02905448, -0.00566903, -0.12401932],
                   [-0.03063667, -0.00570244, -0.12130196],
                   [-0.03221889, -0.00573487, -0.1185846],
                   [-0.03380113, -0.00576583, -0.11586724],
                   [-0.03538342, -0.00579481, -0.11314988],
                   [-0.03696576, -0.00582136, -0.11043253],
                   [-0.03854815, -0.00584498, -0.10771518],
                   [-0.04013062, -0.00586521, -0.10499784],
                   [-0.04171315, -0.00588159, -0.10228052],
                   [-0.04329576, -0.00589367, -0.09956322],
                   [-0.04487846, -0.00590101, -0.09684595],
                   [-0.04646124, -0.00590319, -0.09412873],
                   [-0.04804411, -0.00589978, -0.09141155],
                   [-0.04962707, -0.00589038, -0.08869445],
                   [-0.05121012, -0.00587462, -0.08597742],
                   [-0.05279326, -0.00585212, -0.08326049],
                   [-0.05437648, -0.00582253, -0.08054368],
                   [-0.05595978, -0.00578551, -0.07782701],
                   [-0.05754316, -0.00574074, -0.0751105],
                   [-0.0591266, -0.00568794, -0.07239416],
                   [-0.06071009, -0.00562681, -0.06967804],
                   [-0.06229364, -0.00555712, -0.06696215],
                   [-0.06387722, -0.00547861, -0.06424653],
                   [-0.06546083, -0.00539108, -0.06153119],
                   [-0.06704444, -0.00529434, -0.05881617],
                   [-0.06862806, -0.00518823, -0.0561015],
                   [-0.07021166, -0.00507259, -0.05338722],
                   [-0.07179523, -0.00494731, -0.05067334],
                   [-0.07337874, -0.0048123, -0.04795989],
                   [-0.0749622, -0.00466749, -0.04524692],
                   [-0.07654557, -0.00451282, -0.04253445],
                   [-0.07812886, -0.00434828, -0.0398225],
                   [-0.07971203, -0.00417387, -0.0371111],
                   [-0.08129508, -0.00398963, -0.03440028],
                   [-0.08287799, -0.00379559, -0.03169007],
                   [-0.08446074, -0.00359185, -0.02898047],
                   [-0.08604332, -0.0033785, -0.02627152],
                   [-0.08762573, -0.00315567, -0.02356322],
                   [-0.08920794, -0.00292351, -0.0208556],
                   [-0.09078995, -0.00268219, -0.01814866],
                   [-0.09237175, -0.00243191, -0.01544241],
                   [-0.09395333, -0.00217288, -0.01273686],
                   [-0.09553469, -0.00190535, -0.010032],
                   [-0.09711582, -0.00162958, -0.00732784],
                   [-0.09869672, -0.00134584, -0.00462437],
                   [-0.10027739, -0.00105444, -0.00192157],
                   [-0.10185783, -0.00075571, 0.00078056],
                   [-0.10343804, -0.00044997, 0.00348203],
                   [-0.10501803, -0.00013758, 0.00618288],
                   [-0.1065978, 0.00018108, 0.00888312],
                   [-0.10817736, 0.00050562, 0.01158279],
                   [-0.10975672, 0.00083565, 0.01428191],
                   [-0.11133588, 0.00117074, 0.01698051],
                   [-0.11291487, 0.00151048, 0.01967864],
                   [-0.11449369, 0.00185441, 0.02237633],
                   [-0.11607236, 0.0022021, 0.02507363],
                   [-0.1176509, 0.00255308, 0.02777058],
                   [-0.11922933, 0.00290688, 0.03046723],
                   [-0.12080766, 0.00326303, 0.03316363],
                   [-0.12238591, 0.00362106, 0.03585982],
                   [-0.12396411, 0.00398047, 0.03855586],
                   [-0.12554227, 0.00434077, 0.0412518],
                   [-0.1271204, 0.00470149, 0.0439477]])

N = len(X_orig)

methods = ['manual', 'scipy_cumtrapz', 'torch_cumtrapz']


def normalise(v):
    return v / norm(v, axis=-1, keepdims=True)


def an_orthonormal(x):
    if abs(x[0]) < 1e-20:
        return np.array([1., 0., 0.])
    if abs(x[1]) < 1e-20:
        return np.array([0., 1., 0.])

    X = np.array([x[1], -x[0], 0.])
    return X / norm(X)


def _calculate_tangent_and_curvature_from_midline(X):
    # Distance between vertices
    q = np.linalg.norm(X[1:] - X[:-1], axis=1)
    l = q.sum()
    q = np.r_[q[0], q, q[-1]]

    # Average distances over both neighbours
    spacing = (q[:-1] + q[1:]) / 2
    locs = np.cumsum(spacing)

    # Tangent is normalised gradient of curve
    T = np.gradient(X, locs, axis=0, edge_order=1)
    T_norm = np.linalg.norm(T, axis=-1, keepdims=True)
    T = T / T_norm

    # Curvature is gradient of tangent
    du = l / (N - 1)
    K = np.gradient(T, du, axis=0, edge_order=1)
    K = K / T_norm

    return T, K


def _integrate(Y, Z0, du, method='manual'):
    if method == 'manual':
        Yv = du * (Y[1:] + Y[:-1]) / 2
        Z = np.concatenate([Z0[None, ...], Yv], axis=0).cumsum(axis=0)
    elif method == 'scipy_cumtrapz':
        Z = Z0[None, :] + cumulative_trapezoid(Y, dx=du, axis=0)
        Z = np.r_[Z0[None, :], Z]
    elif method == 'torch_cumtrapz':
        Y = torch.from_numpy(Y)
        Z = Z0[None, :] + torch.cumulative_trapezoid(Y, dx=du, dim=0).numpy()
        Z = np.r_[Z0[None, :], Z]
    return Z


def _calculate_tangent_and_midline_from_curvature(K, T0, X0, method='manual'):
    du = 1 / (N - 1)

    # Integrate curvature to get tangent
    T = _integrate(K, T0, du, method=method)
    # T_norm = np.linalg.norm(T, axis=-1, keepdims=True)
    # T = T / T_norm

    # Integrate tangent to get position
    X = _integrate(T, X0, du, method=method)

    return T, X


def _calculate_bishop_components_from_curvature(K, T):
    """
    Calculate the frame components; M1, M2, their scalar magnitudes, m1, m2, the
    complex representations mc = m1 + j.m2 and from that the scalar curvature and twist.
    """
    start_idx = int(N / 2)
    M1 = np.zeros_like(T)
    M1_tilde = an_orthonormal(T[start_idx])
    M1[start_idx] = normalise(M1_tilde)
    for i in range(start_idx - 1, -1, -1):
        M1_tilde = M1[i + 1] - np.dot(T[i], M1[i + 1]) * T[i]
        M1[i] = normalise(M1_tilde)
    for i in range(start_idx + 1, N):
        M1_tilde = M1[i - 1] - np.dot(T[i], M1[i - 1]) * T[i]
        M1[i] = normalise(M1_tilde)

    # M2 is the remaining orthogonal vector
    M2 = np.cross(T, M1)
    M2 = normalise(M2)

    # Project curvature onto frame
    m1 = np.einsum('ni,ni->n', M1, K)
    m2 = np.einsum('ni,ni->n', M2, K)

    return m1, m2, M1[0]


def _calculate_tangent_and_midline_from_bishop_components(m1, m2, M0, T0, X0, l):
    """
    Convert a Bishop frame representation to recover the full position and components.
    """

    # Initialise the components
    shape = (N, 3)
    X = np.zeros(shape)
    T = np.zeros(shape)
    M1 = np.zeros(shape)
    M2 = np.zeros(shape)
    X[0] = X0
    T[0] = T0
    M1[0] = M0
    M2[0] = np.cross(T[0], M1[0])
    h = l / (N - 1)

    # Calculate the frame components (X/T/M1/M2)
    for i in range(1, N):
        k1 = m1[i]
        k2 = m2[i]

        dTds = k1 * M1[i - 1] + k2 * M2[i - 1]
        dM1ds = -k1 * T[i - 1]
        dM2ds = -k2 * T[i - 1]

        T_tilde = T[i - 1] + h * dTds
        M1_tilde = M1[i - 1] + h * dM1ds
        M2_tilde = M2[i - 1] + h * dM2ds

        X[i] = X[i - 1] + h * T[i - 1]
        T[i] = normalise(T_tilde)
        M1[i] = normalise(M1_tilde)
        M2[i] = normalise(M2_tilde)

    return T, X


def test_round_trip_from_midline():
    X0 = X_orig[0]

    # Calculate tangent and curvature from midline
    T1, K = _calculate_tangent_and_curvature_from_midline(X_orig)
    T0 = T1[0]
    res = {'orig': {'T': T1, 'X': X_orig}}

    # Recompute tangent and midline from curvature using different methods
    for method in methods:
        Tr, Xr = _calculate_tangent_and_midline_from_curvature(K, T0, X0, method=method)
        res[method] = {
            'T': Tr,
            'X': Xr
        }

    fig, axes = plt.subplots(2, 3, figsize=(10, 10))

    for i in range(3):
        for j, TX in enumerate(['T', 'X']):
            ax = axes[j, i]
            ax.set_title(['Midline', 'Tangent'][j] + ' ' + ['x', 'y', 'z'][i])
            for m, r in res.items():
                ax.plot(res[m][TX][:, i], label=m)
            ax.legend()

    fig.tight_layout()

    plt.show()


def test_round_trip_from_curvature():
    x = np.sin(np.linspace(0, np.pi, N))
    # y = np.linspace(0, 10, N)
    y = np.zeros(N)
    # z = np.ones(N)
    z = np.zeros(N)
    K = np.stack([x, y, z], axis=1)

    # K = np.ones((N, 3)) * np.pi
    T0 = np.array([1, 0, 0])
    X0 = np.array([0, 0, 0])

    # Compute tangent and midline from curvature using different methods
    res = {'orig': {'K': K}}
    for method in methods:
        T1, X = _calculate_tangent_and_midline_from_curvature(K, T0, X0, method=method)

        # Recompute tangent and curvature from midline
        T2, K2 = _calculate_tangent_and_curvature_from_midline(X)
        res[method] = {
            'K': K2,
            'T1': T1,
            'T2': T2,
            'X': X
        }

    fig, axes = plt.subplots(4, 3, figsize=(10, 10))

    for i in range(3):
        for j, KTX in enumerate(['K', 'T1', 'T2', 'X']):
            ax = axes[j, i]
            ax.set_title(['Curvature', 'Tangent 1', 'Tangent 2', 'Midline'][j] + ' ' + ['x', 'y', 'z'][i])
            for m, r in res.items():
                if KTX not in r:
                    continue
                ax.plot(res[m][KTX][:, i], label=m)
            ax.legend()

    fig.tight_layout()

    plt.show()


def test_bishop_round_trip_from_midline():
    X0 = X_orig[0]
    l = 1

    # Calculate tangent and curvature from midline
    T1, K1 = _calculate_tangent_and_curvature_from_midline(X_orig)
    T0 = T1[0]

    # Calculate bishop components from curvature
    m1, m2, M0 = _calculate_bishop_components_from_curvature(K1, T1)
    res = {'orig': {'X': X_orig, 'T': T1, 'K': K1, 'm1': m1, 'm2': m2}}

    # Recompute tangent and midline from curvature components
    T2, X2 = _calculate_tangent_and_midline_from_bishop_components(m1, m2, M0, T0, X0, l)
    res['r1'] = {'X': X2, 'T': T2}

    # Recompute the tangent and curvature again to check it matches
    T3, K2 = _calculate_tangent_and_curvature_from_midline(X2)
    res['r2'] = {'T': T3, 'K': K2}

    fig, axes = plt.subplots(4, 3, figsize=(10, 10))

    for j, TX in enumerate(['X', 'T', 'K', 'm1', 'm2']):
        if TX in ['m1', 'm2']:
            if TX == 'm1':
                ax = axes[j, 0]
            elif TX == 'm2':
                ax = axes[j - 1, 1]
            ax.set_title(TX)
            for m, r in res.items():
                if TX not in res[m]:
                    continue
                ax.plot(res[m][TX], label=m)
            ax.legend()
        else:
            for i in range(3):
                ax = axes[j, i]
                ax.set_title(['Midline', 'Tangent', 'Curvature'][j] + ' ' + ['x', 'y', 'z'][i])
                for m, r in res.items():
                    if TX not in res[m]:
                        continue
                    ax.plot(res[m][TX][:, i], label=m)
                ax.legend()

    fig.tight_layout()

    plt.show()


def test_bishop_round_trip_from_curvature():
    m1 = 10 * np.cos(np.linspace(0, 4 * np.pi, N))
    m2 = np.linspace(0, 10, N)
    l = 2

    # K = np.ones((N, 3)) * np.pi
    T0 = np.array([1, 0, 0])
    X0 = np.array([0, 0, 0])
    M0 = an_orthonormal(T0)

    # Compute tangent and midline from curvature using different methods
    T1, X1 = _calculate_tangent_and_midline_from_bishop_components(m1, m2, M0, T0, X0, l)
    m1m2_1 = np.stack([m1, m2], axis=1)
    res = {'orig': {'m1m2': m1m2_1, 'T': T1, 'X': X1, 'k': norm(m1m2_1, axis=-1)}}

    # Recompute tangent and curvature from midline
    T2, K2 = _calculate_tangent_and_curvature_from_midline(X1)

    # Calculate bishop components from curvature
    m12, m22, M02 = _calculate_bishop_components_from_curvature(K2, T2)
    m1m2_2 = np.stack([m12, m22], axis=1)
    res['reconst'] = {'m1m2': m1m2_2, 'T': T2, 'k': norm(K2, axis=-1)}

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(3):
        for j, KTX in enumerate(['m1m2', 'T', 'X']):
            if KTX == 'm1m2' and i == 2:
                continue
            ax = axes[j, i]
            if KTX == 'm1m2':
                if i == 0:
                    ax.set_title('m1')
                else:
                    ax.set_title('m2')
            else:
                ax.set_title(KTX + ' ' + ['x', 'y', 'z'][i])
            for m, r in res.items():
                if KTX not in r:
                    continue
                ax.plot(res[m][KTX][:, i], label=m)
            ax.legend()

    ax = axes[0, 2]
    ax.set_title('k')
    for m, r in res.items():
        ax.plot(res[m]['k'], label=m)
    ax.legend()

    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    # This works :)
    test_round_trip_from_midline()

    # This fails :(
    test_round_trip_from_curvature()

    # These work!
    test_bishop_round_trip_from_midline()
    test_bishop_round_trip_from_curvature()
