import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm, colors
import os
import time
import random
# from skimage.transform import rescale
from numba import jit
import matplotlib.pyplot as plt
from sample import Sample
from scipy import sparse
import _pickle as cPickle
import bz2


class CPM:
    def __init__(self,params=None):
        assert params is not None, "Specify params"
        self.params = params
        self.num_x,self.num_y = None,None
        self.I = None
        self.x_indicies,self.y_indices = None,None
        self.boundary_mask = None

        self.A,self.P,self.A0,self.P0 = None,None,None,None
        self.lambda_P,self.lambda_A = None,None

        self.Moore,self.perim_neighbour = None,None
        self.define_neighbourhood()
        self.sample = Sample(self)
        self.do_steps = self.sample.do_steps

        self.neighbour_options = np.array([[1,0],
                                           [-1,0],
                                           [0,1],
                                           [0,-1]])

    def define_neighbourhood(self):
        i_,j_ = np.meshgrid(np.arange(-1,2),np.arange(-1,2),indexing="ij")
        self.Moore = np.array([i_.ravel(),j_.ravel()]).T
        i_2,j_2 = np.delete(i_.ravel(),4),np.delete(j_.ravel(),4)
        self.perim_neighbour = np.array([i_2,j_2]).T

    def make_grid(self,num_x=300,num_y=300):
        self.num_x,self.num_y = num_x,num_y
        self.I = np.zeros([num_x,num_y]).astype(int)
        self.y_indices, self.x_indicies = np.mgrid[:num_x, :num_y]
        self.boundary_mask = np.ones_like(self.I).astype(bool)
        self.boundary_mask[0],self.boundary_mask[:,0],self.boundary_mask[-1],self.boundary_mask[:,-1] = False,False,False,False

    def generate_cells(self,N_cell_dict):
        self.c_types = []
        for type_i, (Type,n_i) in enumerate(N_cell_dict.items()):
            for i in range(n_i):
                self.c_types += [type_i+1]
        self.n_cells = len(self.c_types)
        self.cell_ids = np.arange(1,self.n_cells+1)
        self.set_cell_params()

    def set_cell_params(self):
        ###index 0 is the medium.
        self.A0 = np.zeros(self.n_cells+1)
        self.P0 = np.zeros(self.n_cells+1)
        self.lambda_A = np.zeros(self.n_cells+1)
        self.lambda_P = np.zeros(self.n_cells+1)
        for i, c_type in enumerate(self.c_types):
            self.A0[i+1] = self.params["A0"][c_type-1]
            self.P0[i+1] = self.params["P0"][c_type-1]
            self.lambda_A[i+1] = self.params["lambda_A"][c_type-1]
            self.lambda_P[i+1] = self.params["lambda_P"][c_type-1]
        self.make_J()


    def make_J(self):
        self.J = np.zeros([self.n_cells+1,self.n_cells+1])
        c_types = np.concatenate(((0,),self.c_types))
        for i in range(len(c_types)):
            for j in range(len(c_types)):
                # if i == 0:
                #     if j!=0:
                #         self.J[i,j] = -self.params["W"][0,self.c_types[j-1]]
                # elif j == 0:
                #     if i!=0:
                #         self.J[i, j] = -self.params["W"][self.c_types[i - 1], 0]
                if i!=j:
                    self.J[i,j] = -self.params["W"][c_types[i],c_types[j]]
        self.get_J_diff()

    def get_J_diff(self):
        self.J_diff = np.expand_dims(self.J,1) - np.expand_dims(self.J,0)

    def make_init(self,init_type="circle",r = 3,spacing = 0.25):
        self.I0 = np.zeros_like(self.I)
        if init_type == "circle":
            X, Y = np.meshgrid(np.arange(self.num_x),np.arange(self.num_y),indexing="ij")
            sq_n_x = int(np.ceil(np.sqrt(self.n_cells))) - 1
            sq_n_y = (int(np.ceil(np.sqrt(self.n_cells))))/2
            x_mid, y_mid = int(self.num_x / 2), int(self.num_y / 2)
            grid_spacing_x = np.arange(-sq_n_x,sq_n_x+1)*(r+spacing)
            grid_spacing_y = np.arange(-sq_n_y,sq_n_y+1)*(r+spacing)*np.sqrt(2)
            x0s,y0s = x_mid + grid_spacing_x, y_mid + grid_spacing_y
            X0,Y0 = np.meshgrid(x0s,y0s,indexing="ij")
            X0,Y0 = np.concatenate([X0[::2,::2].ravel(),X0[1::2,1::2].ravel()]),np.concatenate([Y0[::2,::2].ravel(),Y0[1::2,1::2].ravel()])
            X0 += np.random.uniform(0,0.01,X0.shape)
            Y0 += np.random.uniform(0,0.01,Y0.shape)
            dist_to_mid = (X0-x_mid)**2 + (X0-y_mid)**2
            # grid_choice = np.arange(X0.size)
            grid_choice = np.argsort(dist_to_mid)
            k = 0
            cell_index = np.arange(self.n_cells)
            random.shuffle(cell_index)
            while k < self.n_cells:
                x0,y0 = X0[grid_choice[k]],Y0[grid_choice[k]]
                cll_r = np.sqrt(self.A0[k+1]/np.pi)*0.8
                self.I0[(X-x0+0.5)**2 + (Y-y0+0.5)**2 <= cll_r**2] = cell_index[k]+1
                k+=1

            #
            # grid_choice = grid_choice[:self.n_cells]
            # for k in range(self.n_cells):
            #     x0,y0 = X0[grid_choice[k]],Y0[grid_choice[k]]
            #     cll_r = np.sqrt(self.A0[k+1]/np.pi)*0.8
            #     self.I0[(X-x0+0.5)**2 + (Y-y0+0.5)**2 <= cll_r**2] = k+1
        self.I = self.I0.copy()
        self.assign_AP()

    def assign_AP(self):
        self.A = np.zeros(self.n_cells+1,dtype=int)
        self.P = np.zeros(self.n_cells+1,dtype=int)
        for cll in self.cell_ids:
            self.P[cll],self.A[cll] = self.get_perimeter_and_area(self.I,cll)

    def get_perimeter_and_area(self,I,s):
        """Moore neighbourhood for perimeter"""
        M = I==s
        PI = np.sum(np.array([M!=np.roll(np.roll(M,i,axis=0),j,axis=1) for i,j in self.perim_neighbour]),axis=0)
        P = np.sum(PI*M)
        A = np.sum(M)
        return P,A

    def initialize(self,J0,n_initialise_steps=10000):
        J = self.J.copy()
        lambda_P = self.lambda_P.copy()
        self.lambda_P[:] = np.mean(self.lambda_P)
        self.J = np.zeros_like(self.J)
        self.J[1:,1:] = J0
        self.J = self.J*(1-np.eye(self.J.shape[0]))
        self.get_J_diff()
        self.sample.n_steps = n_initialise_steps
        self.sample.do_steps()
        self.J = J.copy()
        self.lambda_P = lambda_P.copy()
        self.get_J_diff()
        print("initialized")


    def simulate(self,n_step,n_save,initialize=True,J0=None,n_initialise_steps=10000):
        # self.get_xy_clls(self.I)
        if initialize:
            self.initialize(J0,n_initialise_steps)
        self.n_step = n_step
        self.skip = int(n_step/n_save)
        self.sample.n_steps = self.skip
        self.t = np.arange(n_step)
        self.t_save = self.t[::self.skip]
        self.n_save = len(self.t_save)
        self.I_save = np.zeros((self.n_save+1,self.num_x,self.num_y),dtype=int)
        n_steps = int(n_step/self.skip)
        self.I_save[0] = self.I.copy()
        for i in range(n_steps):
            self.sample.do_steps()
            self.I_save[i+1] = self.I.copy()
            print("%.1f"%(100*(i/n_steps)))

    def save_simulation(self,dir_path,name):
        self.I_save_sparse = [None]*len(self.I_save)
        for i, I in enumerate(self.I_save):
            self.I_save_sparse[i] = sparse.csr_matrix(I)
        with bz2.BZ2File(dir_path + "/" + name +  '.pbz2', 'w') as f:
            cPickle.dump(self.I_save_sparse, f)





    def generate_image(self,I, res = 8,col_dict={"E":"red","T":"blue","X":"green"},background=np.array([0,0,0,0.6])):
        I_scale = np.repeat(np.repeat(I, res, axis=0), res, axis=1)
        Im = np.zeros([I.shape[0] * res, I.shape[1] * res, 4]).astype(float)
        Im[:, :, :] = background

        for j in range(1, self.n_cells + 1):
            cll_mask = I_scale == j
            cll_type = self.c_types[j-1]
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

    def get_perimeter_elements(self,I):
        """Only VN neighbours"""
        PI = np.sum(np.array([I!=np.roll(np.roll(I,i,axis=0),j,axis=1) for i,j in self.neighbour_options]),axis=0)
        # return (PI!=0)*self.boundary_mask
        return (PI!=0)

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

    def get_xy_clls(self,I):
        """Only VN neighbours r.e. the D_a """
        self.PE = np.sum(np.array([I!=np.roll(np.roll(I,i,axis=0),j,axis=1) for i,j in self.neighbour_options]),axis=0)
        x_clls, y_clls = np.where((self.get_perimeter_elements(I) != 0)*(self.boundary_mask))
        self.xy_cells = set([])
        for i in range(x_clls.size):
            self.xy_cells.add((x_clls[i], y_clls[i]))
        self.xy_cells = np.array(tuple(self.xy_cells))
        self.nxy_cells = self.xy_cells.shape[0]


