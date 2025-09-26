import os
import numpy as np
from scipy.fftpack import idctn
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg
import torch
from tqdm import tqdm
from scipy.stats import norm
from math import ceil

epsilon = 1e-2

def psi(a):
    return np.where(a > 0, 12.0, 3.0)

def psi_quantile(a, sigma, ps=np.array([0.1, 0.25, 0.5, 0.75, 0.9, 1]), 
                 values=np.array([20.0, 16.0, 12.0, 8.0, 4.0, epsilon]),
                 rng: np.random.Generator = None):
    rng.shuffle(values)
    z_edges = norm.ppf(ps)
    edges = z_edges * sigma
    labels = np.digitize(a, edges)
    return values[labels]

def make_coeff(setting, rng, coeff_type='default'):
    
    if coeff_type == 'default':
        return psi(GRF_DCT(**setting, rng=rng))
    
    elif coeff_type == 'hard1':
        return psi_quantile(*GRF_DCT(**setting, rng=rng, return_with_sigma=True), rng=rng)
    
    elif coeff_type == 'hard2':
        coarse_n = 16
        s = setting['s']
        assert s%coarse_n == 0
        coeff = np.full((s, s), fill_value=epsilon, dtype=float)

        b = s // coarse_n
        total_cells = coarse_n * coarse_n

        levels  = np.array([20.0, 16.0, 12.0, 8.0, 4.0], dtype=float)
        n_patches = np.array([ceil(total_cells*(3/32)), ceil(total_cells*(1/32)), ceil(total_cells*(1/32)), ceil(total_cells*(1/32)), ceil(total_cells*(1/32))], dtype=int)
        assert n_patches.sum() <= total_cells

        chosen = rng.choice(total_cells, size=n_patches.sum(), replace=False)

        start = 0
        for val, cnt in zip(levels, n_patches):
            ids = chosen[start:start+cnt]
            start += cnt
            ii = ids // coarse_n
            jj = ids % coarse_n

            for i, j in zip(ii, jj):
                r0, r1 = i*b, (i+1)*b
                c0, c1 = j*b, (j+1)*b
                coeff[r0:r1, c0:c1] = val

        return coeff
    
    elif coeff_type == 'hard3':
        band_width = 8
        s = setting['s']
        assert s%band_width == 0

        new_setting = {**setting, 's':s//band_width, 'd':1, 'tau':10*setting['tau']}
        line = psi_quantile(*GRF_DCT(**new_setting, rng=rng, return_with_sigma=True), rng=rng)
        row = np.repeat(line, band_width)

        if len(row) > s:
            row = row[:s]
        elif len(row) < s:
            row = np.pad(row, (0, s - len(row)), mode='edge')
        
        coeff = np.tile(row, (s, 1))
        return coeff
    
    elif coeff_type == 'hard4':
        band_width = 8
        s = setting['s']
        assert s%band_width == 0

        new_setting = {**setting, 's':s//band_width, 'd':1, 'tau':10*setting['tau']}
        line = psi_quantile(*GRF_DCT(**new_setting, rng=rng, return_with_sigma=True), rng=rng)
        row = np.repeat(line, band_width)

        if len(row) > s:
            row = row[:s]
        elif len(row) < s:
            row = np.pad(row, (0, s - len(row)), mode='edge')
        
        coeff = np.tile(row, (s, 1)).T
        return coeff


def generate_data(setting, num_data, file, seed=0):
    master = np.random.SeedSequence(seed + hash(file) % (2**32))    
    boundary = setting.pop('boundary', 'ZD')
    coeff_type = setting.pop('coeff_type', 'default')

    if boundary == 'ZD' or boundary == 'ARD1':
        child_seeds = master.spawn(num_data)
        rngs = [np.random.Generator(np.random.PCG64(cs)) for cs in child_seeds]

        F = np.ones((setting['s'], setting['s']))
        boundary_value = np.zeros((setting['s'], setting['s']))

        C_list, U_list = [], []
        for i in tqdm(range(num_data), desc=f'Generating {file}'):
            rng = rngs[i]
            GRF_sample = make_coeff(setting, rng, coeff_type)
            C_list.append(GRF_sample)
            result = solve_darcy_2d(GRF_sample, F, boundary_value=boundary_value)
            U_list.append(result)

    elif boundary == 'ARD2':
        child_seeds = master.spawn(2*num_data)
        rngs = [np.random.Generator(np.random.PCG64(cs)) for cs in child_seeds]

        C_list, U_list = [], []
        for i in tqdm(range(num_data), desc=f'Generating {file}'):
            GRF_sample = make_coeff(setting, rngs[2*i], coeff_type)
            F = GRF_DCT(s=setting['s'], tau=3, alpha=3, d=2,
                        fully_normalized=True, rng=rngs[2*i+1]) + 1
            boundary_value = np.zeros((setting['s'], setting['s']))
            C_list.append([GRF_sample, F])
            result = solve_darcy_2d(GRF_sample, F, boundary_value=boundary_value)
            U_list.append([result])
    
    elif boundary == 'ARD3':
        child_seeds = master.spawn(3*num_data)
        rngs = [np.random.Generator(np.random.PCG64(cs)) for cs in child_seeds]

        C_list, U_list = [], []
        for i in tqdm(range(num_data), desc=f'Generating {file}'):
            GRF_sample = make_coeff(setting, rngs[3*i], coeff_type)
            F = GRF_DCT(s=setting['s'], tau=3, alpha=3, d=2, fully_normalized=True, rng=rngs[3*i+1]) + 1
            boundary_value = GRF_DCT(s=setting['s'], tau=3, alpha=3, d=2, fully_normalized=True, rng=rngs[3*i+2])
            boundary_value[1:-1, 1:-1] = 0
            C_list.append([GRF_sample, F, boundary_value])
            result = solve_darcy_2d(GRF_sample, F, boundary_value=boundary_value)
            U_list.append([result])

    elif boundary == 'ARD4':
        child_seeds = master.spawn(2*num_data)
        rngs = [np.random.Generator(np.random.PCG64(cs)) for cs in child_seeds]

        C_list, U_list = [], []
        for i in tqdm(range(num_data), desc=f'Generating {file}'):
            # Random coefficient and forcing
            GRF_sample = make_coeff(setting, rngs[2*i], coeff_type)
            F = GRF_DCT(s=setting['s'], tau=3, alpha=3, d=2,
                        fully_normalized=True, rng=rngs[2*i+1]) + 1

            # Boundary setup
            s = setting['s']
            boundary_value = np.zeros((s, s))
            boundary_flux = np.zeros((s, s))
            flux_mask = np.zeros((s, s))

            # Dirichlet on x=0 and x=1
            boundary_value[:, 0] = 0.0   # left side
            boundary_value[:, -1] = 1.0  # right side

            # Neumann on y=0 and y=1 → handled via flux_mask
            flux_mask[0, :] = 1
            flux_mask[-1, :] = 1
            boundary_flux[0, :] = 0.0
            boundary_flux[-1, :] = 0.0

            # Save inputs (2 channels: coeff, F)
            C_list.append([GRF_sample, F])

            # Solve with mixed BCs
            result = solve_darcy_2d(GRF_sample, F,
                                    boundary_value=boundary_value,
                                    boundary_flux=boundary_flux,
                                    flux_mask=flux_mask)
            U_list.append([result])

    
    C = torch.from_numpy(np.array(C_list))
    U = torch.from_numpy(np.array(U_list))
    data = {'x': C, 'y': U}
    torch.save(data, file)
    print(f'{file} saved')
        



def GRF_DCT(s: int, tau: float, alpha: float, d: int = 2, fully_normalized: bool = True, mode = None,
            rng: np.random.Generator = None, return_with_sigma: bool = False):

    # [1] Build λ
    k = [np.arange(s) for _ in range(d)]
    K = np.array(np.meshgrid(*k, indexing='ij'))  # shape : (d, *([s]*d))
    sqrt_λ = (np.pi**2 * np.sum(K**2, axis=0) + tau**2)**(-alpha/2)
    if fully_normalized:
        sqrt_λ = tau**(alpha - d/2) * sqrt_λ
    
    # variance
    lam = sqrt_λ**2

    # [2] truncation mask
    if mode is not None:
        slicer = tuple(slice(0, mode) for _ in range(d))
        trunc_mask = np.zeros_like(lam, dtype=bool)
        trunc_mask[slicer] = True
    else:
        trunc_mask = np.ones_like(lam, dtype=bool)

    dc_idx = (0,)*d
    trunc_mask[dc_idx] = False

    sigma2_avg = lam[trunc_mask].sum() / (s**d)
    sigma_avg = float(np.sqrt(sigma2_avg))

    # [3] Sample ξ
    if rng is None:
        rng = np.random.default_rng()
    ξ = rng.normal(0.0, 1.0, size=[s]*d)

    # [4] Construct KL expansion using DCT
    spec = sqrt_λ * ξ
    spec[dc_idx] = 0  # for zero-mean
    
    if mode is not None:
        trunc = np.zeros_like(spec)
        trunc[slicer] = spec[slicer]
        result = idctn(trunc, type=2, norm='ortho')
    else:
        result = idctn(spec, type=2, norm='ortho')

    if return_with_sigma:
        return result, sigma_avg
    else:
        return result


def solve_darcy_2d(coeff, F, boundary_value=None, boundary_flux=None, flux_mask=None):
    s = coeff.shape[0]
    b = F[1:-1, 1:-1].ravel()
    
    rows, cols, data = [], [], []

    def flatten_idx(i, j):
        return i*(s-2) + j
    
    scale = (s-1)**2

    if boundary_value is None:
        boundary_value = np.zeros_like(coeff)
    if flux_mask is None:
        flux_mask = np.zeros_like(coeff)
    
    for i in range(1, s-1):
        for j in range(1, s-1):
            idx = flatten_idx(i-1, j-1)
            a_nx = 0.5 * (coeff[i, j] + coeff[i-1, j])
            a_px = 0.5 * (coeff[i, j] + coeff[i+1, j])
            a_ny = 0.5 * (coeff[i, j] + coeff[i, j-1])
            a_py = 0.5 * (coeff[i, j] + coeff[i, j+1])
            
            diag = 0

            for (di, dj, a) in [(-1, 0, a_nx), (1, 0, a_px), (0, -1, a_ny), (0, 1, a_py)]:
                ii = i+di; jj = j+dj
                if 0 < ii < s-1 and 0 < jj < s-1:
                    rows.append(idx)
                    cols.append(flatten_idx(ii-1, jj-1))
                    data.append(-a*scale)
                    diag = diag + a*scale
                else:
                    if flux_mask[ii, jj] == 0:
                        b[idx] += a*scale * boundary_value[ii, jj]
                        diag = diag + a*scale
                    else:
                        b[idx] += (s-1) * boundary_flux[ii, jj]
            
            rows.append(idx)
            cols.append(idx)
            data.append(diag)

    A = coo_matrix((data, (rows, cols)), shape=((s-2)**2, (s-2)**2)).tocsr()
    
    x, info = cg(A, b, rtol=1e-8, maxiter=50000)
    if info != 0:
        raise RuntimeError(f"CG did not converge (info={info})")
    
    U = boundary_value.copy()
    U[1:-1,1:-1] = x.reshape(s-2, s-2)
    
    if boundary_flux is not None:
        a_nx = 0.5 * (coeff[0, 1:-1] + coeff[1, 1:-1])
        a_px = 0.5 * (coeff[-1, 1:-1] + coeff[-2, 1:-1])
        a_ny = 0.5 * (coeff[1:-1, 0] + coeff[1:-1, 1])
        a_py = 0.5 * (coeff[1:-1, -1] + coeff[1:-1, -2])

        U[0, 1:-1] = U[1, 1:-1] - boundary_flux[0, 1:-1] / (a_nx * (s-1))
        U[-1, 1:-1] = U[-2, 1:-1] + boundary_flux[-1, 1:-1] / (a_px * (s-1))
        U[1:-1, 0] = U[1:-1, 1] - boundary_flux[1:-1, 0] / (a_ny * (s-1))
        U[1:-1, -1] = U[1:-1, -2] + boundary_flux[1:-1, -1] / (a_py * (s-1))
        
        U[0, 0] = 0.5 * (U[1, 0] + U[0, 1])
        U[-1, 0] = 0.5 * (U[-2, 0] + U[-1, 1])
        U[0, -1] = 0.5 * (U[1, -1] + U[0, -2])
        U[-1, -1] = 0.5 * (U[-2, -1] + U[-1, -2])
        
    return U