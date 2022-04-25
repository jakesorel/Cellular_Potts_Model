import numpy as np
from numba import jit

class Zmasks:
    def __init__(self):
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

        # self.z1_masks = np.flip(self.z1_masks,axis=0)
        # self.z2_masks = np.flip(self.z2_masks,axis=0)
        # self.z3_masks = np.flip(self.z3_masks,axis=0)
        self.z_masks = np.concatenate((self.z1_masks,self.z2_masks,self.z3_masks))
        # self.z_masks = np.concatenate((np.flip(self.z1_masks,axis=0),np.flip(self.z2_masks,axis=0),np.flip(self.z3_masks,axis=0)))

        i_,j_ = np.meshgrid(np.arange(-1,2),np.arange(-1,2),indexing="ij")
        self.Moore = np.array([i_.ravel(),j_.ravel()]).T
        i_2,j_2 = np.delete(i_.ravel(),4),np.delete(j_.ravel(),4)
        self.perim_neighbour = np.array([i_2,j_2]).T

        self.get_dP_masks()
        # self.primes = np.array([[2,3,5],[7,11,13],[17,19,23]])
        self.primes = 2**np.arange(9).reshape(3,3)
        self.hashes = np.sum(self.z_masks*self.primes,axis=(1,2))



    def r0(self, x):
        return x

    def r90(self, x):
        return np.flip(x.T, axis=1)

    def r180(self, x):
        return np.flip(np.flip(x, axis=0), axis=1)

    def r270(self, x):
        return np.flip(x.T, axis=0)


    def get_dP_masks(self):
        def getPA(masks):
            P = []
            n = int(masks.shape[0]/2)
            for i in range(n):
                p,a = self.get_perimeter_and_area_not_periodic(masks[i],1)
                p2,a2 = self.get_perimeter_and_area_not_periodic(masks[n+i],1)
                P.append(p2-p)
            return np.concatenate([np.array(P),-np.array(P)]) #check. May need to flip the sign

        self.z1_masks_dP = getPA(self.z1_masks)
        self.z2_masks_dP = getPA(self.z2_masks)
        self.z3_masks_dP = getPA(self.z3_masks)
        self.dP_z = np.concatenate((self.z1_masks_dP,self.z2_masks_dP,self.z3_masks_dP))

    def get_perimeter_and_area_not_periodic(self,I,s):
        M = I == s
        PI = np.sum(np.array([M!=np.roll(np.roll(M,i,axis=0),j,axis=1) for i,j in self.perim_neighbour]),axis=0)
        P = np.sum(PI[1:-1,1:-1])
        A = np.sum(M)
        return P,A

