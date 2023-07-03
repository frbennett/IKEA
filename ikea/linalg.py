import numpy as np
import scipy.linalg as sla
from sklearn.utils.extmath import randomized_svd
import dask.array as da 

def power_iteration(A, Omega, power_iter = 3):
    Y = A @ Omega
    for q in range(power_iter):
        Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y)
    return Q

def rtsvd(A, rank, power_iter = 3):
    Omega = np.random.randn(A.shape[1], rank+1)
    Q = power_iteration(A, Omega, power_iter)
    B = Q.T @ A
    u_tilde, s, v = np.linalg.svd(B, full_matrices = 0)
    u = Q @ u_tilde
    value = s.sum()/(s.sum()+((A.shape[1]-rank)*s.min()))
    u = u[:, :rank]
    v = v[:rank]
    s = s[:rank]
    return u, s, v, value

def tsvd(a,rank):
    u, s, v = np.linalg.svd(a, full_matrices=True, compute_uv=True, hermitian=True)
    total = s.sum()
    u = u[:, :rank]
    v = v[:rank]
    s = s[:rank]
    value = s.sum()/total
    return u, s, v, value

def tinv(a, rank, type='svd', power_iter=3):
    print('Inverse type set to ', type)

    if type == 'tsvd':
        u, s, v, value = tsvd(a, rank)
        pinverse = v.T @ np.diag(s**-1) @ u.T
        print('Approx variance recovered after truncation ', value)
    if type == 'rtsvd':
        u, s, v, value = rtsvd(a, rank, power_iter = power_iter)
        pinverse = v.T @ np.diag(s**-1) @ u.T
        print('Approx variance recovered after rtsvd truncation ', value)
    if type == 'svd':
        pinverse, svd_rank = sla.pinvh(a, return_rank=True)
        print('Rank from full SVD = ', svd_rank)
    
    return pinverse 

def pseudo_inverse(del_D, alpha, Cd, nEnsemble, dLength, mLength, type='svd'):
    if type == 'svd':
        Cdd = (del_D@del_D.T)/(nEnsemble-1)
        K = Cdd + alpha*Cd
        Kinv, svd_rank = sla.pinvh(K, return_rank=True)
    #    print('Rank : ', svd_rank)
    

    if type == 'subspace' :
 #      Efficient subspace inversion
        N = del_D.shape[1]
        D = del_D/np.sqrt(N-1)
        F = alpha * Cd
        Ud, Wd, Vd = np.linalg.svd(D, full_matrices=False, compute_uv=True, hermitian=False)

        r = Wd.size
        Ir = np.diag(np.ones(r))
        X = (np.diag(Wd**-1)@Ud.T@F@Ud@np.diag(Wd**-1))
        Zx, Gamma, ZxT = np.linalg.svd(X)
        Kinv = Ud@np.diag(Wd**-1)@Zx@(np.diag(np.diag(Ir+Gamma)**-1))@(Ud@np.diag(Wd**-1)@Zx).T

    if type == 'rsvd' :
        N = del_D.shape[1]
        random_state=None
        rank = min(del_D.shape)
        print('rank ', rank)
        K =  (del_D@del_D.T)/(N-1) + alpha * Cd
        u, s, v = randomized_svd(K,rank,random_state=None)
        u = u[:, :rank]
        v = v[:rank]
        s = s[:rank]
        Kinv = v.T @ np.diag(s**-1) @ u.T

    if type == 'tsvd' :
        N = del_D.shape[1]
        rank = min(del_D.shape)
        print('rank ', rank)
        K = (del_D@del_D.T)/(N-1) + alpha * Cd
        u, s, v = np.linalg.svd(K, full_matrices=True, compute_uv=True, hermitian=True)
        total = s.sum()
        u = u[:, :rank]
        v = v[:rank]
        s = s[:rank]
        value = s.sum()/total
        print('recovered variance after truncation ', value)
        Kinv = v.T @ np.diag(s**-1) @ u.T

    return Kinv

def dask_inverse(M, Cmd, Duc, D, del_D, phi, alpha, Ne):
    M_da = da.from_array(M, chunks='auto')
    Cmd_da = da.from_array(Cmd, chunks='auto')
    Duc_da = da.from_array(Duc, chunks=(500,Ne))
    D_da = da.from_array(D, chunks=(500,Ne))
    rand_phi_da = da.from_array(phi, chunks=500)
                
    Ud, Wd, Vd = np.linalg.svd(del_D, full_matrices=False, compute_uv=True, hermitian=False)
    Ud = da.from_array(Ud, chunks=(500,Ne))
    Wd = da.from_array(Wd, chunks='auto')
                
    Binv = np.diag(Wd**(-2)) 
                # aCd = (Ne-1) * alpha * phi.mean(axis=1)**2
    aCd = (Ne-1) * alpha * rand_phi_da**2

    AinvUd = ((aCd**(-1))*Ud.T).T
                #bracket = Binv + Ud.T@Ainv@Ud
    bracket = Binv + Ud.T@AinvUd
    bracketinv = np.linalg.inv(bracket)

    Kinv = (Ne-1) * (np.diag(aCd**(-1)) - AinvUd@bracketinv@AinvUd.T)
    M_update_da = M_da+Cmd_da@Kinv@(Duc_da-D_da) 

    M_update = M_update_da.compute() 
    del M_update_da
    return M_update


def efast_inverse(M, Cmd, Duc, D, del_D, rand_phi, alpha, Ne):
    Ud, Wd, Vd = np.linalg.svd(del_D, full_matrices=False, compute_uv=True, hermitian=False)
    Binv = np.diag(Wd**(-2)) 
    # aCd = (Ne-1) * alpha * phi.mean(axis=1)**2
    aCd = (Ne-1) * alpha * rand_phi**2
    # Ainv = np.diag(aCd**(-1))
    AinvUd = ((aCd**(-1))*Ud.T).T
    #bracket = Binv + Ud.T@Ainv@Ud
    bracket = Binv + Ud.T@AinvUd
    bracketinv = np.linalg.inv(bracket)
    #Kinv = (Ne-1) * (Ainv - Ainv@Ud@bracketinv@Ud.T@Ainv)
    Kinv = (Ne-1) * (np.diag(aCd**(-1)) - AinvUd@bracketinv@AinvUd.T)
    M_update = M+Cmd@Kinv@(Duc-D) 
    return M_update
