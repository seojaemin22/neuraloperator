import os
import numpy as np
from scipy.fftpack import idctn
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg
import torch
from tqdm import tqdm

def generate_data(setting, num_data, file, seed=0):
    master = np.random.SeedSequence(seed + hash(file) % (2**32))    
    boundary = setting.pop('boundary', 'ZD')
    if boundary == 'ZD':
        child_seeds = master.spawn(num_data)
        rngs = [np.random.Generator(np.random.PCG64(cs)) for cs in child_seeds]

        F = np.ones((setting['s'], setting['s']))
        boundary_value = np.zeros((setting['s'], setting['s']))

        C_list, U_list = [], []
        for i in tqdm(range(num_data), desc=f'Generating {file}'):
            rng = rngs[i]
            GRF_sample = psi(GRF_DCT(**setting, rng=rng))
            C_list.append(GRF_sample)
            result = solve_darcy_2d(GRF_sample, F, boundary_value=boundary_value)
            U_list.append(result)

    elif boundary == 'ARD1':
        child_seeds = master.spawn(3*num_data)
        rngs = [np.random.Generator(np.random.PCG64(cs)) for cs in child_seeds]

        C_list, U_list = [], []
        for i in tqdm(range(num_data), desc=f'Generating {file}'):
            GRF_sample = psi(GRF_DCT(**setting, rng=rngs[3*i]))
            F = GRF_DCT(s=setting['s'], tau=3, alpha=3, d=2, fully_normalized=True, rng=rngs[3*i+1]) + 1
            boundary_value = GRF_DCT(s=setting['s'], tau=3, alpha=3, d=2, fully_normalized=True, rng=rngs[3*i+2])
            boundary_value[1:-1, 1:-1] = 0
            C_list.append(GRF_sample)
            result = solve_darcy_2d(GRF_sample, F, boundary_value=boundary_value)
            U_list.append(result)

    elif boundary == 'ARD2':
        child_seeds = master.spawn(2*num_data)
        rngs = [np.random.Generator(np.random.PCG64(cs)) for cs in child_seeds]

        C_list, U_list = [], []
        for i in tqdm(range(num_data), desc=f'Generating {file}'):
            GRF_sample = psi(GRF_DCT(**setting, rng=rngs[2*i]))
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
            GRF_sample = psi(GRF_DCT(**setting, rng=rngs[3*i]))
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
            GRF_sample = psi(GRF_DCT(**setting, rng=rngs[2*i]))
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
            rng: np.random.Generator = None):

    # [1] Build λ
    k = [np.arange(s) for _ in range(d)]
    K = np.array(np.meshgrid(*k, indexing='ij'))  # shape : (d, *([s]*d))
    sqrt_λ = (np.pi**2 * np.sum(K**2, axis=0) + tau**2)**(-alpha/2)
    if fully_normalized:
        sqrt_λ = tau**(alpha - d/2) * sqrt_λ
    
    # [2] Sample ξ
    ξ = rng.normal(0.0, 1.0, size=[s]*d)

    # [3] Construct KL expansion using DCT
    spec = sqrt_λ * ξ
    spec[[0]*d] = 0  # for zero-mean
    
    if mode is not None:
        trunc = np.zeros_like(spec)
        slicer = tuple(slice(0, mode) for _ in range(d))
        trunc[slicer] = spec[slicer]
        result = idctn(trunc, type=2, norm='ortho')
    else:
        result = idctn(spec, type=2, norm='ortho')

    return result

def psi(a):
    return np.where(a > 0, 12.0, 3.0)



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