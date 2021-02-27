import numpy as np

def get_normal(x1, a, x):
        
        nth = 1e-4
        
        if np.linalg.norm(x-a) < nth or np.linalg.norm(x+a) < nth:    
            n = np.cross(x, x1)
            fig = True
        else:
            n = np.cross(a, x)
            fig = False
        
        n = n / np.linalg.norm(n)
        
        return n, fig
    
def cart_to_spherical(vec):
    
    r = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    theta = np.arctan2(vec[1], vec[0])
    phi = np.arctan2(vec[2], np.sqrt(vec[0]**2 + vec[1]**2))
    
    return r, theta, phi

def gram_schmidt(mat):
    
    assert mat.shape[0] == mat.shape[1]
    
    basis = np.zeros_like(mat)
    
    basis[:, 0] = mat[:, 0]/np.linalg.norm(mat[:, 0])
    
    for i in range(1, mat.shape[0]):
        v = mat[:, i]
        U = basis[:, :i]
        proj_coeff = np.matmul(U.T, v)
        proj = np.matmul(U, proj_coeff)
        v = v - proj
        
        if np.linalg.norm(v) < 2.2204e-16:
            return
        
        v = v / np.linalg.norm(v)
        basis[:, i] = v
        
    return basis

def estimate_z(dZ, edges, Z1):

    P = dZ.shape[1] + 1
    F = dZ.shape[0]

    Z = np.zeros((F, P))
    Z[:, 0] = Z1

    for i in range(P-1):
        Z[:, edges[1, i]] = Z[:, edges[0, i]] - dZ[:, i]

    return Z