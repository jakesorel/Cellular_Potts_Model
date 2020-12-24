import matplotlib.pyplot as plt
from CPM.CPM import CPM
import matplotlib.gridspec as gridspec
import numpy as np

cpm = CPM()


def add_border(mask,ncol,nrow):
    new_mask = np.ones((nrow*5,ncol*5))*np.nan
    k = 0
    for i in range(nrow):
        for j in range(ncol):
            new_mask[1+i*5:4+i*5,1+j*5:4+j*5] = mask[k]
            k+=1
    return new_mask

fig, ax = plt.subplots()
ax.set_axis_off()
ax.imshow(add_border(cpm.z1_masks,4,8))
fig.savefig("z1_masks.pdf",dpi=300)

fig, ax = plt.subplots()
ax.set_axis_off()
ax.imshow(add_border(cpm.z2_masks,4,8))
fig.savefig("z2_masks.pdf",dpi=300)

fig, ax = plt.subplots()
ax.set_axis_off()
ax.imshow(add_border(cpm.z3_masks,4,6))
fig.savefig("z3_masks.pdf",dpi=300)

VN_neighbours = np.zeros((3,3))
VN_neighbours[1,1] = np.nan
VN_neighbours[1,2],VN_neighbours[2,1],VN_neighbours[1,0],VN_neighbours[0,1] = 1,1,1,1
fig, ax = plt.subplots()
ax.set_axis_off()
ax.imshow(VN_neighbours,cmap=plt.cm.plasma,vmin=0,vmax=2)
fig.savefig("VN.pdf",dpi=300)

M_neighbours = np.ones((3,3))
M_neighbours[1,1] = np.nan
fig, ax = plt.subplots()
ax.set_axis_off()
ax.imshow(M_neighbours,cmap=plt.cm.plasma,vmax=2,vmin=0)
fig.savefig("M_neigh.pdf",dpi=300)