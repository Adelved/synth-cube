
#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.ndimage


def normalize(function):
    return (function - function.min() ) / (function.max()- function.min())

def guassian2D(dim,mux=1,muy=1,sigmax=1,sigmay=1):
    x = np.arange(dim)
    y = np.arange(dim)
    gx = (1/np.sqrt(np.pi*sigmax))*np.exp((-(x-mux)**2/(2*sigmax**2)))
    gy = (1/np.sqrt(np.pi*sigmay))*np.exp((-(y-muy)**2/(2*sigmay**2)))
    
    gx = gx.repeat(dim).reshape((dim,dim))
    gy = gy.repeat(dim).reshape((dim,dim)).T
    g = normalize(gx*gy)
    return g 


def new_fill_above(faultview):
    new = np.zeros_like(faultview)
    for i,row in enumerate(faultview):
        try:
            ind = np.where(row == 1)[0][0]
            new[i,ind:] = 1
        except:
            continue
    return new

def new_fill_below(faultview):
    new = np.zeros_like(faultview)
    for i,row in enumerate(faultview):
        try:
            ind = np.where(row == 1)[0][0]
            new[i,:ind] = 1
        except:
            new[i,:] = 1
    return new
    
def stich_volumes(volume,shifted_volume,above,below):
    return volume*below + shifted_volume*above

class Cube:
    def __init__(self,dim):
        self.dim=dim
        self.init_seis()
        self.init_fault()
        
    def init_seis(self,vmin=-1,vmax=1):
        seis=np.zeros((self.dim,self.dim,self.dim))
        refl = np.random.normal(vmin,vmax,size=self.dim).repeat(self.dim).reshape(self.dim,self.dim)
        for i in range(self.dim):
            seis[:,i,:] = refl
        self.seis = seis
    def init_fault(self):
        self.fault = np.zeros((self.dim,self.dim,self.dim))

    def plot_slices(self,location=0):
        fig,axs = plt.subplots(1,3,figsize=(15,15))
        axs[0].imshow(self.seis[:,location,:],cmap="gray")
        axs[1].imshow(self.seis[:,:,location],cmap="gray")
        axs[2].imshow(self.seis[location,:,:],cmap="gray")

    #input: a set of tuples
    def random_topology(self,num_gaussian,min_smoothing,max_smoothing):
        topology=0
        for i in range(num_gaussian):
            topology+=guassian2D(self.dim,
            np.random.randint(self.dim),
            np.random.randint(self.dim),
            np.random.randint(min_smoothing,max_smoothing),
            np.random.randint(min_smoothing,max_smoothing))
        return topology
    
    def fold_with_gaussian(self,num_gaussian,min_smoothing=30,max_smoothing=100):
        topology = self.random_topology(num_gaussian,min_smoothing,max_smoothing)
        for iline in range(self.seis.shape[0]):
            for i in range(topology.shape[0]):
                self.seis[:,iline,:][:,i:i+1]=sp.ndimage.interpolation.shift(self.seis[:,iline,:][:,i:i+1],(-topology[:,iline][i],0),cval=0)
        return self.seis
    


vol = Cube(128)
vol.fold_with_gaussian(10)
vol.plot_slices(50)









# %%
