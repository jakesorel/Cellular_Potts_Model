import numpy as np
from numba import jit
from zmasks import Zmasks


"""
Would need to include the b_cells segment from the previous code, which requires work. 
"""

class Sample:
    def __init__(self,cpm,n_steps=100):
        self.cpm = cpm
        self.zmasks = Zmasks()
        self.nx,self.ny = 100,100
        self.T = self.cpm.params["T"]
        self.n_steps = n_steps

    def do_step(self):
        self.cpm.I,self.cpm.A,self.cpm.P,self.cpm.xy_cells,self.cpm.nxy_cells = do_step(self.cpm.I, self.cpm.num_x, self.cpm.num_y, self.zmasks.z_masks, self.zmasks.dP_z, self.cpm.A, self.cpm.P, self.cpm.lambda_A, self.cpm.lambda_P, self.cpm.A0, self.cpm.P0, self.cpm.J_diff, self.T,self.zmasks.primes,self.zmasks.hashes,self.cpm.xy_cells,self.cpm.nxy_cells)

    def do_steps(self):
        self.cpm.I,self.cpm.A,self.cpm.P,self.cpm.xy_cells,self.cpm.nxy_cells = do_steps(self.n_steps,self.cpm.I, self.cpm.num_x, self.cpm.num_y, self.zmasks.z_masks, self.zmasks.dP_z, self.cpm.A, self.cpm.P, self.cpm.lambda_A, self.cpm.lambda_P, self.cpm.A0, self.cpm.P0, self.cpm.J_diff, self.T,self.zmasks.primes,self.zmasks.hashes,self.cpm.xy_cells,self.cpm.nxy_cells)


@jit(nopython=True)
def do_step(I,num_x,num_y,z_masks,dP_z,A,P,lambda_A,lambda_P,A0,P0,J_diff,T,primes,hashes,xy_cells,nxy_cells):
    i, j, s, s2,i_xycll = pick_pixel(I,xy_cells,nxy_cells,num_x,num_y)
    dH_1,dH_2 = 0,0
    Na = get_Na(I, i, j)
    # z, Na = get_z(I, i, j, s)
    # z2, Na2 = get_z(I, i, j, s2)

    mask_id_1,mask_id_2 = 0,0
    I2 = I.copy()
    I2[i,j] = s2
    dP_1 = 0
    dP_2 = 0
    dA_1 = 0
    dA_2 = 0
    if s != 0:
        mask_id_1 = get_mask_id(Na==s, primes, hashes)
        if mask_id_1 != -1:
            dP_1 = dP_z[mask_id_1]
            dA_1 = -1
            dH_1 = get_dH(s,dP_1,dA_1,A,P,lambda_A,lambda_P,A0,P0)
    if s2 != 0:
        mask_id_2 = get_mask_id(Na==s2, primes, hashes)
        if mask_id_2 != -1:
            dP_2 = dP_z[mask_id_2]
            dA_2 = 1
            dH_2 = get_dH(s2,dP_2,dA_2,A,P,lambda_A,lambda_P,A0,P0)
    dJ = get_dJ(J_diff,s, s2,Na)
    dH = dH_1 + dH_2 + dJ

    I_new = I.copy()
    changed = False
    if (mask_id_1 != -1)*(mask_id_2 != -1):
        if dH <= 0:
            I_new[i, j] = s2
            A[s] += dA_1
            A[s2] += dA_2
            P[s] += dP_1
            P[s2] += dP_2
            changed = True
        else:
            if np.random.random() < np.exp(-dH / T):
                I_new[i, j] = s2
                A[s] += dA_1
                A[s2] += dA_2
                P[s] += dP_1
                P[s2] += dP_2
                changed = True
    if changed:
        if (s2==0):
            xy_cells,nxy_cells = remove_from_xy_cells(xy_cells,i_xycll,nxy_cells)
        if (s==0):
            xy_cells,nxy_cells = add_xy_cells(xy_cells,i,j,nxy_cells)
    return I_new,A,P,xy_cells,nxy_cells

@jit(nopython=True)
def do_steps(n_steps,I,num_x,num_y,z_masks,dP_z,A,P,lambda_A,lambda_P,A0,P0,J_diff,T,primes,hashes,xy_cells,nxy_cells):
    for i in range(n_steps):
        I,A,P,xy_cells,nxy_cells = do_step(I,num_x,num_y,z_masks,dP_z,A,P,lambda_A,lambda_P,A0,P0,J_diff,T,primes,hashes,xy_cells,nxy_cells)
    return I,A,P,xy_cells,nxy_cells

@jit(nopython=True)
def H(A, P, lambda_A, lambda_P, A0, P0):
    return lambda_A * (A - A0) ** 2 + lambda_P * (P - P0) ** 2

@jit(nopython=True)
def get_dH(s,dP,dA,A,P,lambda_A,lambda_P,A0,P0):
    """if s == 0: then extension; if s2 == 0 then retraction"""
    dH = H(A[s]+dA, P[s]+dP, lambda_A[s], lambda_P[s], A0[s], P0[s])
    dH -= H(A[s], P[s], lambda_A[s], lambda_P[s], A0[s], P0[s])
    return dH




@jit(nopython=True)
def pick_pixel(I,xy_cells,nxy_cells,nx,ny):
    picked = False
    while picked is False:
        i_xycll = int(np.random.random()*nxy_cells)
        i,j = xy_cells[i_xycll]
        if not ((i * j == 0) or (((i - nx + 1) * (j - ny + 1)) == 0)):
            s = I[i, j]
            s2 = get_s2(I, i, j, nx, ny)
            if s!=s2:
                picked = True
    return i,j,s,s2,i_xycll

@jit(nopython=True)
def remove_from_xy_cells(xy_cells,i_xycll,nxy_cells):
    xy_cells = np.row_stack((xy_cells[:i_xycll],xy_cells[i_xycll+1:]))
    nxy_cells -= 1
    return xy_cells,nxy_cells

@jit(nopython=True)
def add_xy_cells(xy_cells,i,j,nxy_cells):
    xy_cells= np.row_stack((xy_cells,np.expand_dims(np.array((i,j)),0)))
    nxy_cells += 1
    return xy_cells,nxy_cells

# @jit(nopython=True)
# def find_mask(z_masks, s, Na, z,dP_z):
#     mask_id = -1
#     cont = True
#     dP = 0
#     if (z != 0)*(z!=4):
#         zi_masks = z_masks[(z-1)*32:(z)*32]
#         n_zi_masks = len(zi_masks)
#         the_mask = Na ==s
#         mask_id = (z-1)*32
#         k = 0
#         while cont:
#             boolean = np.array_equal(zi_masks[k],the_mask)
#             if boolean:
#                 mask_id += k
#                 dP = dP_z[mask_id]
#                 cont = False
#             k +=1
#             if k >= n_zi_masks:
#                 cont = False
#                 mask_id = -1
#     return mask_id,dP



@jit(nopython=True)
def find_mask(z_masks, s, Na, z,dP_z):
    mask_id = -1
    dP = 0
    the_mask = Na ==s
    cont = True
    k = 0
    while cont:
        if k <88:
            is_eq = np.array_equal(z_masks[k],the_mask)
            if is_eq:
                cont = False
                mask_id = k
                dP = dP_z[mask_id]
            else:
                k += 1
        else:
            cont = False
    return mask_id,dP
#



@jit(nopython=True)
def get_mask_id(the_mask,primes,hashes):
    hash = np.sum(the_mask*primes)
    cont = True
    k = 0
    mask_id = -1
    n_hashes = len(hashes)
    while (cont)and(k<n_hashes):
        if hash == hashes[k]:
            mask_id = k
            cont = False
        else:
            k +=1
    return mask_id

@jit(nopython=True)
def sum_axis12(X):
    n = X.shape[0]
    X_out = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        X_out[i] = np.sum(X[i]) == 9
    return X_out

@jit(nopython=True)
def get_z(I, i, j, s):
    Na = I[i - 1:i + 2, j - 1:j + 2]
    z = 4 - np.count_nonzero(Na.take([1, 3, 5, 7]) - s)
    return z, Na

@jit(nopython=True)
def get_Na(I, i, j):
    Na = I[i - 1:i + 2, j - 1:j + 2]
    return Na

@jit(nopython=True)
def get_s2(I, i, j, num_x, num_y):
    neighbour_options = np.array([[1, 0],
                                  [-1, 0],
                                  [0, 1],
                                  [0, -1]])
    ni = neighbour_options[int(np.random.random() * 4)]
    s2 = I[np.mod(ni[0] + i, num_x), np.mod(ni[1] + j, num_y)]
    return s2



@jit(nopython=True)
def sample_xy_clls(Moore,i,j):
    return Moore + np.array([i, j])



# @jit(nopython=True)
# def get_dJ(J,s, s2,Na,Na2):
#     dJ = 0
#     for k in range(4):
#         dJ += J[s2,Na2.take(k)] - J[s,Na.take(k)]
#     for k in range(5,9):
#         dJ += J[s2,Na2.take(k)] - J[s,Na.take(k)]
#     return dJ


@jit(nopython=True)
def get_dJ(J_diff,s, s2,Na):
    # dJ = 0
    Js2s = J_diff[s2,s]
    dJ = Js2s.take(Na.take([0,1,2,3,5,6,7,8])).sum()
    # for k in range(4):
    #     j = Na.take(k)
    #     dJ += Js2s[j]
    # for k in range(5,9):
    #     j = Na.take(k)
    #     dJ += Js2s[j]
    return dJ

# #
# # @jit(cache=True,nopython=True)
# # def P0(A0,SI):
# #     return np.sqrt(A0) * SI
# I = cpm.I.copy()
# A = cpm.A.copy().astype(np.int64)
# P = cpm.P.copy().astype(np.int64)
# lambda_A = cpm.lambda_A
# lambda_P = cpm.lambda_P
# J = cpm.J
# A0 = cpm.A0
# P0 = cpm.P0
# import time
#
# N = int(1e4)
# skip = int(1e2)
# I_save = np.zeros((len(np.arange(N)[::skip]),100,100))
# t0 = time.time()
# k = 0
# for i in range(int(N)):
#     I,A,P = do_step(I, num_x, num_y, z_masks, dP_z, A, P,  lambda_A, lambda_P, A0, P0, J,15)
#     if (i%skip) == 0:
#         I_save[k] = I.copy()
#         k+=1
# t1 = time.time()
# print(t1-t0)
#
# plt.imshow(I)
# plt.show()
#
# cpm.get_perimeter_and_area(I,24)



@jit(nopython=True)
def equal_arr(A,B):
    return np.array_equal(A,B)