import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm, colors
import os
import time
import random
# from skimage.transform import rescale
from numba import jit
# from line_profiler import LineProfiler
from sim_analysis import Graph

class Cell:
    def __init__(self,id,type):
        self.id = id
        self.type = type
        self.A = []
        self.P = []
        self.An = []
        self.Pn = []
        self.A0 = []
        self.P0 = []
        self.SI = []

        self.centroid_t_ = []

        self.lambd_P = []

        self.cell_pol = False



    #
    # @property
    # def P0(self):
    #     return _P0(self.A0,self.SI)

    @property
    def centroid_t(self):
        return np.array(self.centroid_t_)

    def update(self):
        self.A = self.An
        self.P = self.Pn

class CPM:
    def __init__(self):
        self.num_x, self.num_y = [],[]
        self.I = []
        self.Id = []
        self.T = 1
        self.Ni_mask = np.ones([3,3]).astype(int) * np.nan
        self.Ni_mask[1] = 1
        self.Ni_mask[:,1] = 1
        self.Ni_mask[1,1] = np.nan
        self.mask_b = self.Ni_mask.copy()
        self.mask_b[np.isnan(self.mask_b)] = 0

        self.N_mid_mask = np.ones([3,3])
        self.N_mid_mask[1,1] = np.nan

        rots = [self.r0,self.r90,self.r180,self.r270]

        self.z2_mask = np.zeros([3,3]).astype(bool)
        self.z2_mask[0,1],self.z2_mask[1,0],self.z2_mask[0,0] = True,True,True
        self.z2_mask2 = self.z2_mask.copy()
        self.z2_mask2[0,2] = True
        self.z2_mask3 = self.z2_mask2.T
        self.z2_mask4 = self.z2_mask2.copy()
        self.z2_mask4[2,0] = True
        self.z2_masks = np.concatenate([np.array([rots[i](mask) for i in range(4)]) for mask in [self.z2_mask,self.z2_mask2,self.z2_mask3,self.z2_mask4]])
        self.z2_masks_ = self.z2_masks.copy()
        self.z2_masks_[:,1,1] = True
        self.z2_masks = np.concatenate([self.z2_masks,self.z2_masks_])

        self.z3_mask = np.ones([3, 3]).astype(bool)
        self.z3_mask[:,2] = False
        self.z3_mask[1,1] = False
        self.z3_mask2 = self.z3_mask.copy()
        self.z3_mask2[0,2] = True
        self.z3_mask3 = self.z3_mask2.T
        self.z3_masks = np.concatenate([np.array([rots[i](self.z3_mask) for i in range(4)]),np.array([rots[i](self.z3_mask2) for i in range(4)])])
        self.z3_masks = np.concatenate([np.array([rots[i](mask) for i in range(4)]) for mask in [self.z3_mask,self.z3_mask2,self.z3_mask3]])
        self.z3_masks_ = self.z3_masks.copy()
        self.z3_masks_[:,1,1] = True
        self.z3_masks = np.concatenate([self.z3_masks,self.z3_masks_])


        self.z1_mask = np.zeros([3,3]).astype(bool)
        self.z1_mask[0,1] = True
        self.z1_mask2 = self.z1_mask.copy()
        self.z1_mask2[0,2] = True
        self.z1_mask3 = self.z1_mask2.T
        self.z1_mask4 = self.z1_mask2.copy()
        self.z1_mask4[0,0] = True
        self.z1_masks = np.concatenate([np.array([rots[i](mask) for i in range(4)]) for mask in [self.z1_mask,self.z1_mask2,self.z1_mask3,self.z1_mask4]])
        self.z1_masks_ = self.z1_masks.copy()
        self.z1_masks_[:,1,1] = True
        self.z1_masks = np.concatenate([self.z1_masks,self.z1_masks_])

        self.b_change = []

        # self.J = np.array([[0,0,0],
        #                    [0,8,12],
        #                    [0,12,14]])
        #
        # self.J = np.array([[0,16,16,16],
        #                    [16,8,12,8],
        #                    [16,12,14,12],
        #                    [16,8,12,8]])
        # self.J = self.J*0
        # self.J = self.J*0
        # self.J = -self.J
        scaler = 1.5
        self.A0 = 30 * scaler
        self.P0 = 60
        self.dtA = None
        self.A0_div = None

        # J01 = 1800
        # self.W = np.array([[0,J01,J01],
        #                    [J01,0,9000],
        #                    [J01,9000,0]])
        self.lambd_A = 2
        self.lambd_P = 50
        # self.A0 = 150
        # self.B = 200
        # self.T = 1000

        self.neighbour_options = np.array([[1,0],
                                           [-1,0],
                                           [0,1],
                                           [0,-1]])

        i_,j_ = np.meshgrid(np.arange(-1,2),np.arange(-1,2),indexing="ij")
        self.Moore = np.array([i_.ravel(),j_.ravel()]).T
        i_2,j_2 = np.delete(i_.ravel(),4),np.delete(j_.ravel(),4)
        self.perim_neighbour = np.array([i_2,j_2]).T

        self.chooses2i, self.chooses2j = np.array([[0,1],[1,0],[1,2],[2,1]]).T


        self.boundary_mask = []

        self.dP_z = [np.nan,3,0,-2,np.nan]
        self.get_dP_masks()
        self.get_boundary_update_vals()
        self.Im_save = []


    def r0(self,x):
        return x

    def r90(self,x):
        return np.flip(x.T,axis=1)

    def r180(self,x):
        return np.flip(np.flip(x,axis=0),axis=1)

    def r270(self,x):
        return np.flip(x.T,axis=0)

    def make_grid(self,num_x=300,num_y=300):
        self.num_x,self.num_y = num_x,num_y
        self.I = np.zeros([num_x,num_y]).astype(int)
        self.boundary_mask = np.ones_like(self.I).astype(bool)
        self.boundary_mask[0],self.boundary_mask[:,0],self.boundary_mask[-1],self.boundary_mask[:,-1] = True,True,True,True

    def generate_cells(self,N_cell_dict):
        self.cells = [Cell(0,"M")]
        k = 1
        self.cell_type_list = ["M"]
        for Type,n_i in N_cell_dict.items():
            for i in range(n_i):
                self.cells = self.cells + [Cell(k,Type)]
                k += 1
            self.cell_type_list = self.cell_type_list + [Type]
        self.n_cells = k - 1
        self.cell_ids = set(range(0,self.n_cells+1))
        self.set_A0_P0()

    def set_A0_P0(self,rand = False):
        if rand is False:
            for cll in self.cells:
                cll.A0 = self.A0
                cll.P0 = self.P0
        else:
            for cll in self.cells:
                cll.A0 = np.random.uniform(self.A0/2,self.A0)
                cll.P0 = self.P0

    def set_lambdP(self,lambd_P):
        if type(lambd_P) is float:
            for cll in self.cells:
                cll.lambd_P = lambd_P
        elif type(lambd_P) is np.ndarray:
            for cll in self.cells:
                ctype = self.cell_type_list.index(cll.type)
                cll.lambd_P = lambd_P[ctype]


    def set_cll_growth(self,dtA=0.5):
        self.dtA = dtA

    def get_interaction_energy(self,ci,cj):
        return np.sign(self.W[ci, cj])*np.random.lognormal(np.log(np.sign(self.W[ci, cj])*self.W[ci, cj]),self.sigma[ci,cj])

    def symmetrize(self,X):
        upper_tri = np.triu(X)
        Y = upper_tri + upper_tri.T
        return Y

    def make_J(self,W,sigma=None):
        self.W = W
        if sigma is None:
            self.sigma = np.zeros_like(W)
        else:
            self.sigma = sigma
        J = np.zeros([self.n_cells+1,self.n_cells+1])
        for i in range(self.n_cells+1):
            for j in range(self.n_cells+1):
                if i!=j:
                    ci,cj = self.cell_type_list.index(self.cells[i].type),self.cell_type_list.index(self.cells[j].type)
                    J[i,j] = self.get_interaction_energy(ci,cj)
                else:
                    J[i,j] = 0
        J = self.symmetrize(J)
        self.J = J



    def make_init(self,type="circle",r = 3,spacing = 0.25):
        self.I0 = np.zeros_like(self.I)
        if type is "circle":
            X, Y = np.meshgrid(np.arange(self.num_x),np.arange(self.num_y),indexing="ij")
            sq_n_x = int(np.ceil(np.sqrt(self.n_cells))) - 1
            sq_n_y = (int(np.ceil(np.sqrt(self.n_cells))))/2
            x_mid, y_mid = int(self.num_x / 2), int(self.num_y / 2)
            grid_spacing_x = np.arange(-sq_n_x,sq_n_x+1)*(r+spacing)
            grid_spacing_y = np.arange(-sq_n_y,sq_n_y+1)*(r+spacing)*np.sqrt(2)
            x0s,y0s = x_mid + grid_spacing_x, y_mid + grid_spacing_y
            X0,Y0 = np.meshgrid(x0s,y0s,indexing="ij")
            X0,Y0 = np.concatenate([X0[::2,::2].ravel(),X0[1::2,1::2].ravel()]),np.concatenate([Y0[::2,::2].ravel(),Y0[1::2,1::2].ravel()])
            grid_choice = np.arange(X0.size)
            random.shuffle(grid_choice)
            grid_choice = grid_choice[:self.n_cells]
            for k in range(self.n_cells):
                x0,y0 = X0[grid_choice[k]],Y0[grid_choice[k]]
                cll_r = np.sqrt(self.cells[k+1].A0/np.pi)*0.8
                self.I0[(X-x0+0.5)**2 + (Y-y0+0.5)**2 <= cll_r**2] = k+1
        self.I = self.I0.copy()
        self.A = np.zeros_like(self.I)
        self.P = np.zeros_like(self.I)
        self.assign_AP()

    def assign_AP(self):
        cell_ids = list(self.cell_ids)[1:]
        for cll in cell_ids:
            P,A = self.get_perimeter_and_area(self.I,cll)
            self.cells[cll].A, self.cells[cll].P = A, P

    def plot_I(self):
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(np.flip(self.I.T,axis=0))
        fig.show()

    # def get_z(self,I,i,j,s):
    #     Na = I[i-1:i+2,j-1:j+2]
    #     Ni = Na[self.chooses2i, self.chooses2j]
    #     z = sum(Ni==s)
    #     return z,Na
    #
    def get_z(self,I,i,j,s):
        return _get_z(I,i,j,s)
        # Na = I[i-1:i+2,j-1:j+2]
        # Ni = Na[self.chooses2i, self.chooses2j]
        # z = sum(Ni==s)
        # return z,Na

    #
    # def LA(self,s,Na,z):
    #     if z == 0:
    #         return False
    #     if z == 4:
    #         return False
    #     if z == 1:
    #         return ((Na == s) == self.z1_masks).all(axis=1).all(axis=1)
    #     if z == 2:
    #         return ((Na == s) == self.z2_masks).all(axis=1).all(axis=1) #or should be geq?
    #
    #     if z == 3:
    #         return ((Na == s) == self.z3_masks).all(axis=1).all(axis=1) #or should be geq?


    def LA(self,s,Na,z):
        return _LA(self.z1_masks,self.z2_masks,self.z3_masks,s,Na,z)

    def get_dP_masks(self):
        def getPA(masks):
            P = []
            n = int(masks.shape[0]/2)
            for i in range(n):
                p,a = self.get_perimeter_and_area_not_periodic(masks[i],1)
                p2,a2 = self.get_perimeter_and_area_not_periodic(masks[n+i],1)
                P.append(p2-p)
            return np.concatenate([np.array(P),-np.array(P)]) #check. May need to flip the sign

        z1_masks_dP = getPA(self.z1_masks)
        z2_masks_dP = getPA(self.z2_masks)
        z3_masks_dP = getPA(self.z3_masks)
        self.dP_masks = [None,z1_masks_dP,z2_masks_dP,z3_masks_dP,None]

    def get_boundary_update_vals(self):
        """Find self and non-self interactions within the 4 neighbours of the central cell.
        If self-self = -1; If self-non-self = +1"""
        def get_b_change(masks,z):
            ni = int(masks.shape[0]/2)
            b_change = np.array([self.mask_b*(~masks[i]) -  self.mask_b*masks[ni+i] for i in range(ni)]).astype(int)
            b_change[:,1,1] = 4-2*z
            return np.concatenate([b_change, -b_change])
        self.b_change = [None,get_b_change(self.z1_masks,1),get_b_change(self.z2_masks,2),get_b_change(self.z3_masks,3),None]

    def get_perimeter_elements(self,I):
        """Only VN neighbours"""
        PI = np.sum(np.array([I!=np.roll(np.roll(I,i,axis=0),j,axis=1) for i,j in self.neighbour_options]),axis=0)
        # return (PI!=0)*self.boundary_mask
        return (PI!=0)

    def get_perimeter_elements_Moore(self,I):
        """Only VN neighbours"""
        PI = np.sum(np.array([I!=np.roll(np.roll(I,i,axis=0),j,axis=1) for i,j in self.perim_neighbour]),axis=0)
        # return (PI!=0)*self.boundary_mask
        return (PI!=0)


    def get_perimeter_and_area(self,I,s):
        """Moore neighbourhood for perimeter"""
        M = I==s
        PI = np.sum(np.array([M!=np.roll(np.roll(M,i,axis=0),j,axis=1) for i,j in self.perim_neighbour]),axis=0)
        P = np.sum(PI*M)
        A = np.sum(M)
        return P,A

    def get_perimeter_and_area_not_periodic(self,I,s):
        M = I == s
        PI = np.sum(np.array([M!=np.roll(np.roll(M,i,axis=0),j,axis=1) for i,j in self.perim_neighbour]),axis=0)
        P = np.sum(PI[1:-1,1:-1])
        A = np.sum(M)
        return P,A

    def get_perimeter_and_area_het(self,I,s,s2):
        """Check!!!!"""
        M = I==s
        M2 = I==s2
        P = np.sum(np.array([M*np.roll(M2,i,axis=j) for i,j in self.perim_neighbour]))
        A = np.sum(M)
        return P,A

    def get_area(self,I,s):
        return np.sum(I==s)

    # def get_dJ(self,I, I2, i, j, s, s2):
    #     Na = I[i - 1:i + 2, j - 1:j + 2]
    #     Na2 = I2[i - 1:i + 2, j - 1:j + 2]
    #     dJ = (self.J[s2,Na2.take([0,1,2,3,5,6,7,8])] - self.J[s,Na.take([0,1,2,3,5,6,7,8])]).sum()
    #     return dJ

    def get_dJ(self,I, I2, i, j, s, s2):
        return _get_dJ(self.J, I, I2, i, j, s, s2)

    def get_dJ_pol(self,I, I2, i, j, s, s2,ii,jj):
        if self.cells[s].cell_pol:
            cx,cy = self.cells[s].centroid
            perp_pol = self.cells[s].perp_pol
            side = np.sign((i-cx)*perp_pol[1] - (j-cy)*perp_pol[0])
            if side < 0:
                p = 1
            else:
                p = 0
        else:
            p = 1
        if self.cells[s2].cell_pol:
            c2x,c2y = self.cells[s2].centroid
            perp_pol = self.cells[s2].perp_pol
            side = np.sign((ii-c2x)*perp_pol[1] - (jj - c2y)*perp_pol[0])
            if side < 0:
                p2 = 1
            else:
                p2 = 0
        else:
            p2 = 1
        return _get_dJ(self.J, I, I2, i, j, s, s2)*p*p2


    def H(self,A,P,A0,P0,cll):
        return _H(A,P,self.lambd_A,cll.lambd_P,A0,P0)


    def get_dH(self,I,i,j,s,s2,z,z2,LA,LA2):
        """if s == 0: then extension; if s2 == 0 then retraction"""
        I2 = I.copy()
        I2[i, j] = s2
        if s == 0:
            dH1 = 0
        else:
            dP, dA = int(self.dP_masks[z][LA]), -1  # Triple check the signs are correct here.
            cell1 = self.cells[s]
            cell1.An, cell1.Pn = cell1.A+dA,cell1.P+dP
            dH1 = self.H(cell1.An,cell1.Pn,cell1.A0,cell1.P0,cell1) - self.H(cell1.A,cell1.P,cell1.A0,cell1.P0,cell1)
        if s2 == 0:
            dH2 = 0
        else:
            dP2,dA2 = int(self.dP_masks[z2][LA2]),+1
            cell2 = self.cells[s2]
            cell2.An, cell2.Pn = cell2.A+dA2,cell2.P+dP2
            dH2 = self.H(cell2.An,cell2.Pn,cell2.A0,cell2.P0,cell2) - self.H(cell2.A,cell2.P,cell2.A0,cell2.P0,cell2)
        dJ = self.get_dJ(I, I2, i, j, s, s2)
        dH = dH1 + dH2 + dJ
        return dH,I2


    def get_dH_pol(self,I,i,j,s,s2,z,z2,LA,LA2,ii,jj):
        """if s == 0: then extension; if s2 == 0 then retraction"""
        I2 = I.copy()
        I2[i, j] = s2
        if s == 0:
            dH1 = 0
        else:
            dP, dA = int(self.dP_masks[z][LA]), -1  # Triple check the signs are correct here.
            cell1 = self.cells[s]
            cell1.An, cell1.Pn = cell1.A+dA,cell1.P+dP
            dH1 = self.H(cell1.An,cell1.Pn,cell1.A0,cell1.P0,cell1) - self.H(cell1.A,cell1.P,cell1.A0,cell1.P0,cell1)
        if s2 == 0:
            dH2 = 0
        else:
            dP2,dA2 = int(self.dP_masks[z2][LA2]),+1
            cell2 = self.cells[s2]
            cell2.An, cell2.Pn = cell2.A+dA2,cell2.P+dP2
            dH2 = self.H(cell2.An,cell2.Pn,cell2.A0,cell2.P0,cell2) - self.H(cell2.A,cell2.P,cell2.A0,cell2.P0,cell2)
        dJ = self.get_dJ_pol(I, I2, i, j, s, s2,ii,jj)
        dH = dH1 + dH2 + dJ
        return dH,I2


    def get_s2(self,I,i,j):
        return _get_s2(I,i,j)

    def get_s2_pol(self,I,i,j):
        return _get_s2_pol(I,i,j)

    def perform_transform(self,dH,I,I2,s,s2,z,LA,i,j):
        if dH <= 0:
            self.cells[s].update()
            self.cells[s2].update()
            self.update_xy_clls(i, j, z, LA,I,I2)
            return I2
        else:
            if np.random.random() < np.exp(-dH / self.T):
                self.cells[s].update()
                self.cells[s2].update()
                self.update_xy_clls(i, j, z, LA, I, I2)
                return I2
            else:
                return I


    def do_step(self,I):
        i, j = random.choice(self.xy_clls_tup)
        #Pick a site i,j, State is s
        s = I[i,j]


        s2 = self.get_s2(I,i,j)
        if s2 == s:
            return I
        elif s==0:
            z2, Na = self.get_z(I,i,j,s2)
            LA2 = self.LA(s2,Na,z2)
            if LA2.any():
                dH, I2 = self.get_dH(I,i,j,s,s2,None,z2,None,LA2)
                # print(LA)
                return self.perform_transform(dH,I,I2,s,s2,None,None,i,j)
            else:
                return I
        elif s2==0:
            z, Na = self.get_z(I,i,j,s)
            LA = self.LA(s,Na,z)
            if LA.any():
                dH, I2 = self.get_dH(I, i, j, s, s2,z,None,LA,None)
                return self.perform_transform(dH,I,I2,s,s2,z,LA,i,j)
            else:
                return I
        else:
            z, Na = self.get_z(I,i,j,s)
            LA = self.LA(s,Na,z)
            if LA.any():
                z2, Na = self.get_z(I, i, j, s2)
                LA2 = self.LA(s2, Na, z2)
                if LA2.any():
                    dH, I2 = self.get_dH(I, i, j, s, s2, z, z2, LA, LA2)
                    return self.perform_transform(dH,I,I2,s,s2,z,LA,i,j)
                else:
                    return I
            else:
                return I


    def do_step_pol(self,I):
        i, j = random.choice(self.xy_clls_tup)
        #Pick a site i,j, State is s
        s = I[i,j]
        s2,(ii,jj) = self.get_s2_pol(I,i,j)
        if s2 == s:
            return I
        elif s==0:
            z2, Na = self.get_z(I,i,j,s2)
            LA2 = self.LA(s2,Na,z2)
            if LA2.any():
                dH, I2 = self.get_dH_pol(I,i,j,s,s2,None,z2,None,LA2,ii,jj)
                # print(LA)
                return self.perform_transform(dH,I,I2,s,s2,None,None,i,j)
            else:
                return I
        elif s2==0:
            z, Na = self.get_z(I,i,j,s)
            LA = self.LA(s,Na,z)
            if LA.any():
                dH, I2 = self.get_dH_pol(I, i, j, s, s2,z,None,LA,None,ii,jj)
                return self.perform_transform(dH,I,I2,s,s2,z,LA,i,j)
            else:
                return I
        else:
            z, Na = self.get_z(I,i,j,s)
            LA = self.LA(s,Na,z)
            if LA.any():
                z2, Na = self.get_z(I, i, j, s2)
                LA2 = self.LA(s2, Na, z2)
                if LA2.any():
                    dH, I2 = self.get_dH_pol(I, i, j, s, s2, z, z2, LA, LA2,ii,jj)
                    return self.perform_transform(dH,I,I2,s,s2,z,LA,i,j)
                else:
                    return I
            else:
                return I



    def get_xy_clls(self,I):
        """Only VN neighbours r.e. the D_a """
        self.PE = np.sum(np.array([I!=np.roll(np.roll(I,i,axis=0),j,axis=1) for i,j in self.neighbour_options]),axis=0)
        x_clls, y_clls = np.where(self.get_perimeter_elements(I) != 0)
        self.n_clls = x_clls.size
        self.xy_clls = set([])
        for i in range(self.n_clls):
            self.xy_clls.add((x_clls[i], y_clls[i]))
        self.xy_clls_tup = tuple(self.xy_clls)


    def update_xy_clls(self,i,j,z,LA,I,I2):
        Pa = self.PE[i-1:i+2,j-1:j+2].copy()
        if z is None:
            z,Na = self.get_z(I,i,j,0)
            b_change = -self.mask_b*(Na!=0) + self.mask_b*(I2[i-1:i+2,j-1:j+2]==0)
            b_change[1,1] = -4+2*z
        else:
            b_change = self.b_change[z][LA]

        self.PE[i-1:i+2,j-1:j+2] = Pa + b_change
        Pa_new = self.PE[i - 1:i + 2, j - 1:j + 2]

        rm_xy_clls = (Pa_new==0)*(Pa!=0)
        add_xy_clls = (Pa_new!=0)*(Pa==0)
        sample_xy_clls = _sample_xy_clls(self.Moore,i,j)
        if sum(sum(rm_xy_clls))!=0:
            self.xy_clls -= set((map(tuple,sample_xy_clls[rm_xy_clls.ravel()])))
        if sum(sum(add_xy_clls))!=0:
            self.xy_clls.update(list(map(tuple,sample_xy_clls[add_xy_clls.ravel()])))
        self.xy_clls_tup = tuple(self.xy_clls)


    def compile_xy_cll_matrix(self):
        xy_clls = list(self.xy_clls)
        B = np.zeros_like(self.I0)
        for i in range(len(xy_clls)):
            B[xy_clls[i]] = 1
        return B

    def polarise(self,I,s,Im,type_id):
        X, Y = np.mgrid[:I.shape[0], :I.shape[1]]
        bes = self.get_perimeter_elements_Moore(I == s) * (I != s)
        b_vals = ((Im == 0) * -1 + (Im != type_id) * (Im != 0) * 1) * bes
        b_n = np.sum(bes)
        if (((np.sum(b_vals<0)/b_n)>0.3)&((np.sum(b_vals>0)/b_n)>0.3))or(self.cells[s].cell_pol):
            xc, yc = np.sum(X * (I == s)) / np.sum(I == s), np.sum(Y * (I == s)) / np.sum(I == s)
            self.cells[s].centroid = np.array([xc,yc])
            polx, poly = np.sum(b_vals * (X * bes - xc)), np.sum(b_vals * (Y * bes - yc))
            pol = np.array([polx, poly])  # points away from medium
            div_plane = np.dot(np.array([[0, -1], [1, 0]]), pol)
            self.cells[s].perp_pol = div_plane
            self.cells[s].cell_pol = True

    def raw_moment(self,data, i_order, j_order):
      nrows, ncols = data.shape
      y_indices, x_indicies = np.mgrid[:nrows, :ncols]
      return (data * x_indicies**i_order * y_indices**j_order).sum()

    def moments_cov(self,data):
        """https://stackoverflow.com/questions/9005659/compute-eigenvectors-of-image-in-python"""
        data_sum = data.sum()
        m10 = self.raw_moment(data, 1, 0)
        m01 = self.raw_moment(data, 0, 1)
        x_centroid = m10 / data_sum
        y_centroid = m01 / data_sum
        u11 = (self.raw_moment(data, 1, 1) - x_centroid * m01) / data_sum
        u20 = (self.raw_moment(data, 2, 0) - x_centroid * m10) / data_sum
        u02 = (self.raw_moment(data, 0, 2) - y_centroid * m01) / data_sum
        cov = np.array([[u20, u11], [u11, u02]])
        return cov,np.array([x_centroid,y_centroid])


    def split_cell(self,X):
        cov, centroid = self.moments_cov(X)
        evals, evecs = np.linalg.eig(cov)
        short_ax = evecs[:,evals.argmin()]
        nrows, ncols = X.shape
        x_indices, y_indices = np.mgrid[:nrows, :ncols]
        grad = short_ax[1]/short_ax[0]
        mask = grad*(x_indices - centroid[0]) > (y_indices - centroid[1])
        mask = mask.T
        cA, cB = mask*X, (~mask)*X
        def remove_discon(I):
            PE = np.sum(np.array([I != np.roll(np.roll(I, i, axis=0), j, axis=1) for i, j in self.neighbour_options]), axis=0)
            return I*(PE!=4)

        return remove_discon(cA),remove_discon(cB)


    def set_div_thresh(self,A0_div=60):
        self.A0_div = 60

    def divide(self,cll,I):
        if cll.A > self.A0_div:
            id = cll.id
            id_new = I.max() + 1
            cA,cB = self.split_cell(I == id)
            I[I==id] = cA[I==id]*id + cB[I==id]*id_new
            new_A0 = cll.A0/2
            cll.A0 = new_A0
            self.cells = self.cells + [Cell(id_new,cll.id)]
            new_cll = self.cells[-1]
            new_cll.A0 = new_A0
            new_cll.SI = self.SI

            cll.P,cll.A = self.get_perimeter_and_area(I, id)
            new_cll.P, new_cll.A = self.get_perimeter_and_area(I, id_new)
            new_cll.type = cll.type
            self.cell_ids.add(id_new)
            self.n_cells = self.n_cells + 1
            J_new = np.zeros([self.n_cells + 1, self.n_cells + 1])
            J_new[:-1,:-1] = self.J
            for i in range(self.n_cells + 1):
                if i != id_new:
                    ci, cj = self.cell_type_list.index(self.cells[i].type), self.cell_type_list.index(
                        self.cells[id].type)
                    J_new[i, id] = self.get_interaction_energy(ci, cj)
                else:
                    J_new[i, id] = 0
            J_new = self.symmetrize(J_new)
            self.J = J_new

            self.get_xy_clls(I)
        return I


    def run_simulation(self,n_steps,n_save,polarise=False):
        I = self.I
        i_save = np.zeros(n_steps).astype(bool)
        i_save[::int(n_steps/n_save)] = True
        n_save = i_save[::int(n_steps/n_save)].size
        I_save = np.zeros([n_save,self.num_x,self.num_y])
        ns = 0
        self.get_xy_clls(I)
        if polarise is True:
            do_step = self.do_step_pol
        else:
            do_step = self.do_step
        if self.A0_div is None:
            for ni in range(n_steps):
                for k in range(len(self.xy_clls_tup)):
                    I = do_step(I)
                if i_save[ni]:
                    print(np.round(ni/n_steps * 100),"%")
                    I_save[ns] = I
                    self.I_save = I_save
                    ns += 1
                if polarise:
                    Im = self.cell_type_matrix(I)
                    if "X" in self.cell_type_list:
                        type_id = self.cell_type_list.index("X")
                        for cll in self.cells[1:]:
                            if cll.type == "X":
                                self.polarise(I, cll.id,Im,type_id)

        else:
            for ni in range(n_steps):
                for k in range(len(self.xy_clls_tup)):
                    I = do_step(I)
                    self.I = I
                for cll in self.cells[1:]:
                    cll.A0 += self.dtA
                    I = self.divide(cll, I)
                    self.I = I
                if i_save[ni]:
                    print(np.round(ni / n_steps * 100), "%")
                    I_save[ns] = I
                    self.I_save = I_save
                    ns += 1
        self.I = I
        self.I_save = I_save
        return self.I

    #
    #
    #
    #
    # def do_steps(self,I):
    #     for ni in range(self.n_steps):
    #         i, j = random.choice(self.xy_clls_tup)
    #         # i,j = list(self.xy_clls)[int(np.random.random()*len(self.xy_clls))]
    #         #Pick a site i,j, State is s
    #         s = I[i,j]
    #
    #         s2 = self.get_s2(I,i,j)
    #         if s2 == s:
    #             I = I
    #         elif s==0:
    #             z2, Na = self.get_z(I,i,j,s2)
    #             LA2 = self.LA(s2,Na,z2)
    #             if LA2.any():
    #                 dH, I2 = self.get_dH(I,i,j,s,s2,None,z2,None,LA2)
    #                 # print(LA)
    #                 dJ = self.get_dJ(I, I2, i, j, s, s2)
    #                 I = self.perform_transform(dH,I,I2,s,s2,None,None,i,j)
    #         elif s2==0:
    #             z, Na = self.get_z(I,i,j,s)
    #             LA = self.LA(s,Na,z)
    #             if LA.any():
    #                 # print(LA)
    #
    #                 dH, I2 = self.get_dH(I, i, j, s, s2,z,None,LA,None)
    #                 dJ = self.get_dJ(I, I2, i, j, s, s2)
    #
    #                 I =  self.perform_transform(dH,I,I2,s,s2,z,LA,i,j)
    #         else:
    #             z, Na = self.get_z(I,i,j,s)
    #             LA = self.LA(s,Na,z)
    #             if LA.any():
    #                 z2, Na = self.get_z(I, i, j, s2)
    #                 LA2 = self.LA(s2, Na, z2)
    #                 if LA2.any():
    #                     # print(LA)
    #                     dH, I2 = self.get_dH(I, i, j, s, s2, z, z2, LA, LA2)
    #                     dJ = self.get_dJ(I, I2, i, j, s, s2)
    #                     I =  self.perform_transform(dH,I,I2,s,s2,z,LA,i,j)
    #     return I
    #
    # def run_simulation_profile(self,n_steps):
    #     I = self.I
    #     self.get_xy_clls(I)
    #     self.n_steps = n_steps
    #     lp = LineProfiler()
    #     lp_wrapper = lp(self.do_steps)
    #     lp_wrapper(self.I0)
    #     lp.print_stats()
    #     return


    def generate_image(self,I, res = 8,col_dict={"E":"red","T":"blue"},background=np.array([0,0,0,0.6])):
        I_scale = np.repeat(np.repeat(I, res, axis=0), res, axis=1)
        Im = np.zeros([I.shape[0] * res, I.shape[1] * res, 4]).astype(float)
        Im[:, :, :] = background

        for j in range(1, self.n_cells + 1):
            cll_mask = I_scale == j
            cll_type = self.cells[j].type
            col_name = col_dict.get(cll_type)
            if type(col_name) is str:
                col = np.array(colors.to_rgba(col_name))
            else:
                col = col_name
            Im[cll_mask] = col
        boundaries = self.get_perimeter_elements(I_scale)
        I_scale[boundaries] = 0
        Im[boundaries] = 0
        return Im

    def generate_image_t(self,res = 8,col_dict={"E":"red","T":"blue"},background=np.array([0,0,0,0.6])):
        """col_dict says what colours each cell type is. Can be RGBA or names"""
        n_save = self.I_save.shape[0]
        Im_save = np.zeros([n_save,self.I_save.shape[1]*res,self.I_save.shape[2]*res,4])
        for ni, I in enumerate(self.I_save):
            Im_save[ni] = self.generate_image(I, res,col_dict,background)
        self.Im_save = Im_save

    def animate(self,file_name=None, dir_name="plots", xlim=None, ylim=None, quiver=False, voronoi=False,
                **kwargs):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

        def animate(i):
            ax1.clear()
            ax1.set(aspect=1)
            ax1.axis('off')
            ax1.imshow(self.Im_save[i])

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, bitrate=1800)

        if file_name is None:
            file_name = "animation %d" % time.time()

        an = animation.FuncAnimation(fig, animate, frames=self.I_save.shape[0], interval=200)
        an.save("%s/%s.mp4" % (dir_name, file_name), writer=writer, dpi=264)



    def get_centroids_t(self):
        nrows, ncols = self.I.shape
        y_indices, x_indicies = np.mgrid[:nrows, :ncols]
        centroids = np.zeros([self.I_save.shape[0],self.n_cells+1,2])
        def raw_moment(data, i_order, j_order):
            return (data * x_indicies ** i_order * y_indices ** j_order).sum()
        def get_centroid(data):
            data_sum = data.sum()
            m10 = raw_moment(data, 1, 0)
            m01 = raw_moment(data, 0, 1)
            y_centroid = m10 / data_sum
            x_centroid = m01 / data_sum
            return np.array([x_centroid, y_centroid])
        for t, I in enumerate(self.I_save):
            for s in np.unique(I).astype(int):
                centroid = get_centroid(I == s)
                centroids[t,s] = centroid
                self.cells[s].centroid_t_.append(centroid)
        self.centroids = centroids


    def get_MSD(self,t0):
        return np.array([np.mean([np.linalg.norm(self.centroids[t0, i] - self.centroids[t, i])
                                  for i in range(1,self.n_cells + 1)])
                         for t in range(t0, self.I_save.shape[0])])

    def get_neighbourhood_index(self):
        """
        see
        https://www.researchgate.net/publication/335071700_Distinct_modes_of_cell_competition_are_governed_by_entropic_and_energetic_properties_of_mixed_cell_populations/link/5d4fffc4299bf1995b75fe72/download
        """
        neighbourhood_percentage = np.zeros([self.centroids.shape[0],self.centroids.shape[1]-1,len(self.cell_type_list)-1])
        masks = []
        for i, type in enumerate(self.cell_type_list[1:]):
            mask = np.array([cell.type == type for cell in self.cells[1:]])
            masks.append(mask)
            n_r = np.array([np.sqrt(cell.A/np.pi) for cell in self.cells[1:]])
            Ri,Rj = np.meshgrid(n_r,n_r,indexing="ij")
            for t, centroids in enumerate(self.centroids):
                X_, Y_ = np.outer(centroids[1:,0],np.ones(centroids.shape[0]-1)),np.outer(centroids[1:,1],np.ones(centroids.shape[0]-1))
                d = np.sqrt((X_-X_.T)**2 + (Y_-Y_.T)**2)
                neighbours = (d<(Ri+Rj))*(d>0)
                neighbourhood_percentage[t,:,i] = np.sum(neighbours*mask,axis=1)/np.sum(neighbours,axis=1)

        self.neighbourhood_percentage = np.array([np.nanmean(neighbourhood_percentage[:,masks[i],i],axis=1) for i in range(len(masks))]).T#ignore isolated cells
        self.shannon_entropy = np.nanmean(-neighbourhood_percentage*np.log2(neighbourhood_percentage),axis=(1,2))*2


    def get_neighbourhood_index2(self):
        neighbourhood_percentage = np.zeros([self.I_save.shape[0],len(self.cells)-1,len(self.cell_type_list)-1])
        neighbourhood_adjacency_matrix = np.zeros([self.I_save.shape[0],len(self.cells)-1,len(self.cells)-1])
        masks = []
        for i, type in enumerate(self.cell_type_list[1:]):
            mask = np.array([cell.type == type for cell in self.cells[1:]])
            masks.append(mask)
            for t, I in enumerate(self.I_save):
                for j, cll in enumerate(self.cells[1:]):
                    cll_mask = I==j+1
                    perim = self.get_perimeter_elements_Moore(cll_mask)
                    neighbours = np.unique(I[perim*(~cll_mask)]).astype(int)
                    neighbours = neighbours[neighbours>0] - 1
                    neighbourhood_percentage[t,j,i] = np.sum(mask[neighbours])/neighbours.size
                    neighbourhood_adjacency_matrix[t,j,neighbours] = 1
        self.neighbourhood_percentage = np.array([np.nanmean(neighbourhood_percentage[:,masks[i],i],axis=1) for i in range(len(masks))]).T#ignore isolated cells
        self.shannon_entropy = np.nanmean(-neighbourhood_percentage*np.log2(neighbourhood_percentage),axis=(1,2))*2
        self.neighbourhood_adjacency_matrix = neighbourhood_adjacency_matrix
        self.cell_type_masks = np.array(masks)




    def cell_type_matrix(self,I):
        Im = np.zeros([I.shape[0], I.shape[1]]).astype(int)
        for j in range(1, self.n_cells + 1):
            cll_mask = I == j
            cll_type_id = self.cell_type_list.index(self.cells[j].type)
            Im[cll_mask] = cll_type_id
        return Im


    def find_subpopulations(self,I):
        """For each cell type, finds the number of subpopulations (i.e. cases where cells or clusters of cells are
        seperated by more than one element-width (including diagonals).

        Returns a list: ES,TS,XEN"""

        clique_sizes = []
        Im = self.cell_type_matrix(I)
        for i in np.arange(1,len(self.cell_type_list)):
            grid = 1 * (Im == i)
            g = Graph()
            clique_sizes.append(g.numIslandsDFS(grid.tolist()))
        return np.array(clique_sizes)

    def find_subpopulations_t(self):
        self.subpopulations_t = np.array([self.find_subpopulations(I) for I in self.I_save])
        return self.subpopulations_t


    def get_pol_stat(self,I):
        Im = self.cell_type_matrix(I)
        nrows, ncols = I.shape
        X, Y = np.mgrid[:nrows, :ncols]
        R_tot = np.array([np.sum((X)*(Im!=0))/np.sum(Im!=0),np.sum((Y)*(Im!=0))/np.sum(Im!=0)])
        p = np.sqrt(np.sum(((X*(Im!=0) - R_tot[0])**2 + (Y*(Im!=0) - R_tot[1])**2)))/np.sum(Im!=0)
        R_i = np.array([np.array([np.sum((X)*(Im==i))/np.sum(Im==i),np.sum((Y)*(Im==i))/np.sum(Im==i)]) for i in range(1, len(self.cell_type_list))])
        n_cell_type = len(self.cell_type_list)-1
        f = np.zeros(n_cell_type)
        for i in range(n_cell_type):
            R_ii = np.roll(R_i,-1,axis=0)
            ri = R_ii[0]
            rj = R_ii[1:]
            if rj.ndim > 1:
                rj = np.mean(rj,axis=0)
            f[i] = np.linalg.norm(ri-rj)/p
        return f

    def get_polt(self):
        polt = np.array([self.get_pol_stat(I) for I in self.I_save])
        self.polt = polt
        return polt


    def get_F_xy_ij(self,I, i, j):
        """Unfinished. Need to include the J term. """
        Fxy = np.zeros([4])
        s = I[i,j]
        cell1 = self.cells[s]
        def dH(I,I2):
            cell1.Pn,cell1.An = self.get_perimeter_and_area(I2,s)
            # dJ = self.get_dJ(I, I2, i, j, s, I2[i,j])
            return self.H(cell1.An, cell1.Pn, cell1.A0, cell1.P0) - self.H(cell1.A, cell1.P, cell1.A0, cell1.P0)
        for k, (di,dj) in enumerate(self.neighbour_options):
            I2 = I.copy()
            I2[i,j] = I2[i+di,j+dj]
            I3 = I.copy()
            I2[i+di,j+dj] = I2[i,j]
            F = -np.sign(di+dj)*(dH(I,I2) - dH(I,I3))
            Fxy[k] = F
        Fx = Fxy[:2].mean()
        Fy = Fxy[2:].mean()
        return Fx,Fy

    def get_F_xy(self,I):
        FX,FY = np.zeros_like(I),np.zeros_like(I)
        for (i,j) in self.xy_clls_tup:
            if I[i,j]!=0:
                Fx,Fy = self.get_F_xy_ij(I,i,j)
                FX[i,j],FY[i,j] = Fx,Fy
        self.FX,self.FY = FX,FY
        return FX, FY

    def smooth_Fxy(self,I, FX,FY):
        X, Y = np.mgrid[-3:4, -3:4]
        r3neigh = X ** 2 + Y ** 2 <= 3 ** 2
        di, dj = X[r3neigh], Y[r3neigh]
        return smooth_Fxy_(I, FX,FY,self.xy_clls_tup,di,dj)

@jit(cache=True,nopython=True)
def _w(I,i,j,di,dj):
    II,JJ = i+di,j+dj
    s = I[i,j]
    same = 0
    for ii,jj in zip(II,JJ):
        same += (I[ii,jj] == s)*1.0
    return same

# @jit(cache=True,nopython=True)
def smooth_Fxy_(I, FX, FY,xy_clls_tup,di,dj):
    FX_smooth,FY_smooth = np.zeros_like(FX).astype(float),np.zeros_like(FY).astype(float)
    for (i,j) in xy_clls_tup:
        s = I[i,j]
        II,JJ = i+di,j+dj
        Nx,Ny = 0,0
        for ii,jj in zip(II,JJ):
            same = (I[ii,jj] == s)
            if same:
                Nx += FX[ii,jj]*_w(I,ii,jj,di,dj)
                Ny += FY[ii,jj]*_w(I,ii,jj,di,dj)
        n = np.array([Nx,Ny])/np.sqrt(Nx**2 + Ny**2)
        F_abs = np.sqrt(FX[i,j]**2 + FY[i,j]**2)
        FX_smooth[i,j],FY_smooth[i,j] = F_abs*n
    return FX_smooth,FY_smooth




@jit(cache=True, nopython=True)
def _LA(z1_masks,z2_masks,z3_masks,s,Na,z):
    if z == 1:
        return sum_axis12((Na == s) == z1_masks)
    if z == 2:
        return sum_axis12((Na == s) == z2_masks) #or should be geq?
    if z == 3:
        return sum_axis12((Na == s) == z3_masks) #or should be geq?
    if z == 4:
        return np.array([False])

@jit(cache=True, nopython=True)
def sum_axis12(X):
    n = X.shape[0]
    X_out = np.zeros(n,dtype=np.bool_)
    for i in range(n):
        X_out[i] = np.sum(X[i])==9
    return X_out



@jit(cache=True, nopython=True)
def _get_z(I,i,j,s):
    Na = I[i-1:i+2,j-1:j+2]
    z = 4-np.count_nonzero(Na.take([1,3,5,7]) - s)
    return z,Na

# @jit(cache=True, nopython=True)
# def _get_s2(I, i, j):
#     opts = np.array(list(set(I[i - 1:i + 2, j - 1:j + 2].take([1, 3, 5, 7]))))
#     return np.random.choice(opts)

@jit(cache=True, nopython=True)
def _get_s2(I, i, j):
    neighbour_options = np.array([[ 1,  0],
                                   [-1,  0],
                                   [ 0,  1],
                                   [ 0, -1]])
    ni = neighbour_options[int(np.random.random()*4)]
    s2 = I[ni[0]+i,ni[1]+j]
    return s2


@jit(cache=True, nopython=True)
def _get_s2_pol(I, i, j):
    neighbour_options = np.array([[ 1,  0],
                                   [-1,  0],
                                   [ 0,  1],
                                   [ 0, -1]])
    ni = neighbour_options[int(np.random.random()*4)]
    s2 = I[ni[0]+i,ni[1]+j]
    return s2,np.array([ni[0]+1,ni[1]+j])



# @jit(cache=True, nopython=True)
# def _get_ij(xy_clls):
#     opts = tuple(xy_clls)
#     return tuple(xy_clls)[int(np.random.random()*len(xy_clls))]

#
# @jit
# def _update_xy_clls(i,j,z,LA,I,I2,b_change,PE,Moore,xy_clls,n_bound):
#     Pa = PE[i-1:i+2,j-1:j+2].copy()
#     # if z is None:
#     #     z,Na = _get_z(I,i,j,0)
#     #     b_change = -Na.take([1,3,5,7])!=0 + I2[i-1:i+2,j-1:j+2].take([1,3,5,7])==0
#     #     b_change = b_change.reshape(3,3)
#     #     b_change[1,1] = -4+2*z
#     # else:
#     # b_change = B_change[LA]
#
#     PE[i-1:i+2,j-1:j+2] = Pa + b_change
#     Pa_new = PE[i - 1:i + 2, j - 1:j + 2]
#
#     rm_xy_clls = ((Pa_new==0)*(Pa!=0)).ravel()
#     n_rm = np.sum(rm_xy_clls)
#     add_xy_clls = ((Pa_new!=0)*(Pa==0)).ravel()
#     n_add = np.sum(add_xy_clls)
#     sample_xy_clls = Moore + np.array([i, j])
#
#     for k, rmi in enumerate(rm_xy_clls):
#         if rmi is True:
#             sample_xy_clls[k]
#
#     if n_rm!=0:
#         print(sample_xy_clls[rm_xy_clls[0]])
#         print(sample_xy_clls[rm_xy_clls[0]].dtype)
#         for k in range(n_rm):
#             if k ==0:
#                 rm_mat = (xy_clls - sample_xy_clls[rm_xy_clls[k]])==0
#             else:
#                 rm_mat = rm_mat + (xy_clls - sample_xy_clls[rm_xy_clls[k]])==0
#         # xy_clls = xy_clls[rm_mat]
#
#     if n_add!=0:
#         xy_add = sample_xy_clls[add_xy_clls]
#
#     # n_clls =
#     # if np.sum(rm_xy_clls)!=0:
#     #     xy_clls =
#     #     xy_clls -= set((map(tuple,sample_xy_clls[rm_xy_clls.ravel()])))
#     # if sum(sum(add_xy_clls))!=0:
#     #     self.xy_clls.update(list(map(tuple,sample_xy_clls[add_xy_clls.ravel()])))
#     return PE,xy_clls

@jit(cache=True, nopython=True)
def _sample_xy_clls(Moore,i,j):
    return Moore + np.array([i, j])

@jit(cache=True, nopython=True)
def _H(A, P, lambda_A, lambda_P, A0, P0):
    return lambda_A * (A - A0) ** 2 + lambda_P * (P - P0) ** 2

# @jit(cache=True, nopython=True)
# def _H(A, P, lambda_A, lambda_P, A0, P0):
#     return lambda_A * (A - A0) ** 2 + lambda_P * P** 2
# #
# @jit(cache=True, nopython=True)
# def _H_ND(A, P, r,p0,A0):
#     r = lambda_A*A0/lambda_P
#     return (A/np.sqrt(A0) - 1) ** 2 + (P/np.sqrt(A0) -p0)** 2
# # #
#
# @jit(cache=True, nopython=True)
# def _H(A, P, lambda_A, lambda_P, A0, P0):
#     return lambda_A * ((A - A0)/A0) ** 2 + lambda_P * ((P - P0)/P0) ** 2
#

@jit(cache=True, nopython=True)
def _get_dJ(J,I, I2, i, j, s, s2):
    Na = I[i - 1:i + 2, j - 1:j + 2].take([0,1,2,3,5,6,7,8])
    Na2 = I2[i - 1:i + 2, j - 1:j + 2].take([0,1,2,3,5,6,7,8])
    dJ = 0
    for k in range(8):
        dJ += J[s2,Na2[k]] - J[s,Na[k]]
    return dJ


@jit(cache=True,nopython=True)
def _P0(A0,SI):
    return np.sqrt(A0) * SI



@jit(nopython=True,cache=True)
def element_cross_abs(X,x):
    Y = np.empty((X.shape[1],X.shape[2]),dtype=X.dtype)
    for i in range(X.shape[1]):
        for j in range(X.shape[2]):
            Y[i,j] = np.sign(X[0,i,j]*x[1] - X[1,i,j]*x[0])
    return Y
