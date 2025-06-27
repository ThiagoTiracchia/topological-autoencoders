import numpy as np
from scipy.spatial.distance import pdist, squareform
from IPython import embed
import torch

def findclose(x, A, tol=1e-5):
    return ((x + tol) >= A) & ((x - tol) <= A)


def lipschitz(dX, dY):
    return np.max(dY / dX)

def lipschitzTorch(dX, dY):
    return torch.max(dY / dX)

def matrix_size_from_condensed(dX):
   # n = len(dX)
   # return int(0.5 * (np.sqrt(8 * n + 1) - 1) + 1)

    # Convertir a tensor de PyTorch si es necesario
    if not isinstance(dX, torch.Tensor):
        dX = torch.tensor(dX, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Caso 1: Matriz cuadrada 2D
    if dX.dim() == 2:
        if dX.shape[0] != dX.shape[1]:
            raise ValueError("La matriz debe ser cuadrada.")
        return dX.shape[0]
    
    # Caso 2: Vector condensado 1D
    elif dX.dim() == 1:
        n_elements = dX.shape[0]
        # Convertimos a tensor antes de aplicar torch.sqrt
        value = 8.0 * float(n_elements) + 1.0
        sqrt_value = torch.sqrt(torch.tensor(value, device=dX.device))
        return int(0.5 * (sqrt_value - 1.0)) + 1
       
    
    else:
        raise ValueError("La entrada debe ser un vector 1D o matriz cuadrada 2D.")
#pareciera que estaba bien la cuenta, lo que ocurra en caso de llamar a esta funcion que antes habia que llamar a eneral_position_distance_matrixTorch

#aramr una matriz y pasarle pdist n*n
#chequear tamaño 
#largo del vector meterlo en la funcion de arriba de  arriba
#ver que devuelva n

def to_condensed_form(i, j, m):
    return m * i + j - ((i + 2) * (i + 1)) // 2.0


def general_position_distance_matrix(X, perturb=1e-7):
    n = len(X)
    Xperturbation = perturb * np.random.rand((n * (n - 1) // 2))
    dX = pdist(X) + Xperturbation
    return dX

def general_position_distance_matrixTorch(X, perturb=1e-7,device='cuda'):
    n = len(X)
    Xperturbation = perturb * torch.rand((n * (n - 1) // 2),device=device)
    dX = torch.pdist(X) + Xperturbation
    return dX

def conematrix(DX, DY, DY_fy, eps):
    n = len(DX)
    m = len(DY)

    D = np.zeros((n + m + 1, n + m + 1))
    D[0:n, 0:n] = DX
    D[n : n + m, n : n + m] = DY

    D[0:n, n : n + m] = DY_fy
    D[n : n + m, 0:n] = DY_fy.T

    R = max(DX.max(), DY_fy.max()) + 1

    D[n + m, n : n + m] = R
    D[n : n + m, n + m] = R

    D[n + m, :n] = eps
    D[:n, n + m] = eps

    return D


def conematrixTorch(DX, DY, DY_fy, eps):
    n = len(DX)
    m = len(DY)
    
    D = torch.zeros((n + m + 1, n + m + 1),device=DY.device)
    
    D[0:n, 0:n] = DX
    D[n : n + m, n : n + m] = DY




    D[0:n, n : n + m] = DY_fy # pincha aca ->>dps  dejo de pinchar pq cambue lo de cambie lo de matrix_from_condensed
    D[n : n + m, 0:n] = DY_fy.T

    R = max(DX.max(), DY_fy.max()) + 1 # ver si correr esto con pytorch es el numero mas grande entre los 2.

    D[n + m, n : n + m] = R
    D[n : n + m, n + m] = R

    D[n + m, :n] = eps
    D[:n, n + m] = eps

    return D




def squareform_torch(vec, force="no", checks=True):
    """
    Convierte entre un vector de distancia condensado y una matriz cuadrada simétrica.
    Compatible con CUDA y equivalente a `scipy.spatial.distance.squareform`.

    Args:
        vec (torch.Tensor): Tensor 1D (vector condensado) o 2D (matriz cuadrada).
        force (str): Si 'tovector' o 'tomatrix', fuerza la conversión en esa dirección.
        checks (bool): Si True, verifica que la entrada sea válida.

    Returns:
        torch.Tensor: Matriz simétrica (si entrada es vector) o vector condensado (si entrada es matriz).
    """
    if vec.dim() not in [1, 2]:
        raise ValueError("El tensor de entrada debe ser 1D (vector) o 2D (matriz).")

    # Conversión de matriz a vector (similar a squareform en SciPy)
    if vec.dim() == 2 or force == "tovector":
        if checks:
            assert vec.shape[0] == vec.shape[1], "La matriz debe ser cuadrada."
            assert torch.allclose(vec, vec.t()), "La matriz debe ser simétrica."
        
        n = vec.shape[0]
        rows, cols = torch.triu_indices(n, n, offset=1)
        return vec[rows, cols]

    # Conversión de vector a matriz (similar a squareform en SciPy)
    elif vec.dim() == 1 or force == "tomatrix":
        if checks:
            num_elements = vec.numel()
            n = int((1 + (1 + 8 * num_elements)**0.5) / 2)
            expected_elements = n * (n - 1) // 2
            assert num_elements == expected_elements, f"Tamaño del vector incorrecto. Esperado {expected_elements}, obtenido {num_elements}."

        n = int((1 + (1 + 8 * vec.numel())**0.5) / 2)
        mat = torch.zeros((n, n), dtype=vec.dtype, device=vec.device)
        rows, cols = torch.triu_indices(n, n, offset=1)
        mat[rows, cols] = vec
        mat[cols, rows] = vec  # Hacer simétrica
        return mat

    else:
        raise ValueError("Argumento 'force' no válido. Usar 'tovector' o 'tomatrix'.")


def format_bars(bars):
    bars = [np.array(b) for b in bars]
    lens = list(map(len, bars))
    for i in range(len(bars)):
        if all(l == 0 for l in lens[i:]):
            bars = bars[:i]
            break
    return bars


def kercoker_bars(dgm, dgmX, dgmY, cone_eps, tol=1e-11):
    """
    Find cokernel and kernel bars in the persistence diagram.
    TODO: optimize
    """
    coker_dgm = [[] for _ in range(len(dgm))]
    ker_dgm = [[] for _ in range(len(dgm))]
    for k in range(len(dgm)):
        for r in dgm[k]:
            b, d = r
            if d > cone_eps + tol:
                # coker
                # b_c = b_y_i
                # d_c = d_y_i
                m = findclose(b, dgmY[k][:, 0], tol) & findclose(d, dgmY[k][:, 1], tol)
                if sum(m):
                    coker_dgm[k].append((b, d))

                # b_c = b_y_i
                # d_c = b_x_j
                if any(findclose(b, dgmY[k][:, 0], tol)) and any(findclose(d, dgmX[k][:, 0], tol)):
                    coker_dgm[k].append((b, d))

                # ker
                if k > 0:
                    # b_c = b_x_i (dim-1)
                    # d_c = d_x_i (dim-1)
                    m = findclose(b, dgmX[k - 1][:, 0], tol) & findclose(d, dgmX[k - 1][:, 1], tol)
                    if sum(m):
                        ker_dgm[k - 1].append((b, d))

                    # b_c = d_y_i (dim-1)
                    # d_c = d_x_j (dim-1)
                    if any(findclose(b, dgmY[k - 1][:, 1], tol)) and any(findclose(d, dgmX[k - 1][:, 1], tol)):
                        ker_dgm[k - 1].append((b, d))

    coker_dgm = format_bars(coker_dgm)
    ker_dgm = format_bars(ker_dgm)
    return coker_dgm, ker_dgm
