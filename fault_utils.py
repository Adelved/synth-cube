
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.ndimage
def normalize(function):
    return (function - function.min() ) / (function.max()- function.min())

g = lambda x,mu,sigma: 1/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2/(2*sigma**2))

def guassian2D(dim,mux=1,muy=1,sigmax=1,sigmay=1):
    x = np.arange(dim)
    y = np.arange(dim)
    gx = (1/np.sqrt(np.pi*sigmax))*np.exp((-(x-mux)**2/(2*sigmax**2)))
    gy = (1/np.sqrt(np.pi*sigmay))*np.exp((-(y-muy)**2/(2*sigmay**2)))
    
    gx = gx.repeat(dim).reshape((dim,dim))
    gy = gy.repeat(dim).reshape((dim,dim)).T
    g = normalize(gx*gy)
    return g 


def partial_fill_above(faultview,depth=100000):
    
    new = np.zeros_like(faultview)
    
    for i,row in enumerate(faultview):
        if i < depth:
            try:
                ind = np.where(row == 1)[0][0]
                new[i,ind:] = 1
            except:
                continue
    return new

def partial_fill_below(faultview,depth=100000):
    new = np.zeros_like(faultview)
    for i,row in enumerate(faultview):
        if i < depth:
            try:
                ind = np.where(row == 1)[0][0]
                new[i,:ind] = 1
            except:
                new[i,:] = 1
        else:
            try:
                ind = np.where(row == 1)[0][0]
                new[i,:] = 1
            except:
                new[i,:] = 1
            
    return new

def shift_volume_down(volume,amount):
    shifted = sp.ndimage.interpolation.shift(volume,(amount,0,0),cval=0,prefilter=False,order=1)
    return shifted

def stich_volumes(volume,shifted_volume,above,below):
    return volume*below + shifted_volume*above




def fault_from_fill(filled_view):
    x = []
    y = []
    for i in range(filled_view.shape[0]):
        try:
            xx = i
            yy = np.where(filled_view[i,:] == 1)[0].min()
            x.append(xx)
            y.append(yy)
        except:
            continue
    
    fview = np.zeros(filled_view.shape)
    fview[x,y] = 1
    return fview

def normalize_with_max(function,maxval):
    return (function - function.min() ) / (maxval - function.min())

def deg_to_rad(deg):
    return (np.pi/180)*deg

def clip_within_bounds(dim,yvals,dip_orientation):
    x,y = 0,0
    if yvals.max() >= dim:
        try:
            if np.cos(dip_orientation) > 0:
                val = np.where(yvals >= dim)[0].min()
                y = yvals[:val]
                x = np.arange(dim)[:val]
            elif np.cos(dip_orientation) < 0:
                val = np.where(yvals >= dim)[0].max()
                print(val)
                y = yvals[val+1:]
                x = np.arange(dim)[val+1:]
        except Exception as e:
            print(e)
            
    else:
        x = np.arange(dim)
        y = yvals

    return x,y

def normal_fault(dim, dip, start_location = 0, return_values=True):

    x = np.arange(dim)
    
    dip_rad = deg_to_rad(dip)
    y = x * np.cos(dip_rad)
    
    y = ((normalize_with_max(y,dim)) * dim + start_location).astype(int)
    
    xx,yy = clip_within_bounds(dim,y,dip_rad)
    
    view = np.zeros((dim,dim))
    view[xx,yy] = 1
    view = partial_fill_above(view)
    
    
    if return_values == True:
        fault = fault_from_fill(view)
        xx,yy = np.where(fault == 1)
        return xx,yy

    else:
        return fault_from_fill(view)

def listric_fault(dim,depth_horizontal = 1.1, start_location = 0, return_values=True):
    if depth_horizontal <= 1:
        depth_horizontal =1.1
    if start_location > dim:
        start_location = dim
    
    x = np.arange(dim)
    y = (x**2 / (depth_horizontal*dim - x))
    y = ((normalize_with_max(y,dim)) * dim + start_location).astype(int)
    
    xx,yy = clip_within_bounds(dim,y,1)
    
    view = np.zeros((dim,dim))
    view[xx,yy] = 1
    view = partial_fill_above(view)
    
    if return_values == True:
        fault = fault_from_fill(view)
        xx,yy = np.where(fault == 1)
        return xx,yy
    else:
        return fault_from_fill(view)

class Cube:
        
    def init_seis(self,vmin=-1,vmax=1):
        seis=np.zeros((self.dim,self.dim,self.dim))
        refl = np.random.normal(vmin,vmax,size=self.dim).repeat(self.dim).reshape(self.dim,self.dim)
        for i in range(self.dim):
            seis[:,i,:] = refl
        self.seis = seis

    def init_fault(self):
        self.fault = np.zeros((self.dim,self.dim,self.dim))

    def plot_seis_slices(self,location=0):
        fig,axs = plt.subplots(1,3,figsize=(15,15))
        axs[0].imshow(self.seis[:,location,:],cmap="gray")
        axs[1].imshow(self.seis[:,:,location],cmap="gray")
        axs[2].imshow(self.seis[location,:,:],cmap="gray")

    
    def plot_fault_slices(self,location=0):
        fig,axs = plt.subplots(1,3,figsize=(15,15))
        axs[0].imshow(self.fault[:,location,:],cmap="gray")
        axs[1].imshow(self.fault[:,:,location],cmap="gray")
        axs[2].imshow(self.fault[location,:,:],cmap="gray")

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
        

    def single_plane_fault(self,
    dip=45,
    position=50,
    throw=5,
    orientation=10,
    strike_type="linear",
    inplace=True):

        x,y = normal_fault(self.dim,dip,position)
        fault = np.zeros((self.dim,self.dim))

        fault[x,y] = 1
        if strike_type == "linear":
            strike = (normalize(np.arange(self.dim))*np.random.randint(0,orientation)).astype(int)
        elif strike_type == "curved":
            strike = (normalize(g(np.arange(self.dim),self.dim//2,self.dim))*orientation).astype(int)
        elif strike_type=="composite":
            strike =  ((normalize(g(np.arange(self.dim),self.dim//2,self.dim))*orientation).astype(int) + 
            (normalize(np.arange(self.dim))*np.random.randint(0,orientation)).astype(int) )

        above = np.zeros((self.dim,self.dim,self.dim))
        below = np.zeros((self.dim,self.dim,self.dim))
        fvol = np.zeros((self.dim,self.dim,self.dim))

        for i in range(self.dim):
            fvol[:,i,:] = sp.ndimage.interpolation.shift(fault,(0,strike[i]),cval=0,prefilter=False,order=1)
            above[:,i,:] = partial_fill_above(fvol[:,i,:])
            below[:,i,:] = partial_fill_below(fvol[:,i,:])

        seisvol = stich_volumes(self.seis,shift_volume_down(self.seis,throw),above,below)
        self.seis = seisvol

        if len(np.unique(self.fault)) == 1:
            self.fault = fvol
        else:
            self.fault = stich_volumes(self.fault,shift_volume_down(self.fault,throw),above,below)
            self.fault += fvol
        

    def single_listric_fault(self,dip=45,
    depth_horizontal=1.1,
    position=50,
    throw=5,
    orientation=10,
    strike_type="linear",
    inplace=True):

        x,y = listric_fault(self.dim,depth_horizontal,position)
        fault = np.zeros((self.dim,self.dim))
        fault[x,y] = 1

        if strike_type == "linear":
            strike = (normalize(np.arange(self.dim))*np.random.randint(0,orientation)).astype(int)
        elif strike_type == "curved":
            strike = (normalize(g(np.arange(self.dim),self.dim//2,self.dim))*orientation).astype(int)
        elif strike_type=="composite":
            strike =  ((normalize(g(np.arange(self.dim),self.dim//2,self.dim))*orientation).astype(int) + 
            (normalize(np.arange(self.dim))*np.random.randint(0,orientation)).astype(int) )

        above = np.zeros((self.dim,self.dim,self.dim))
        below = np.zeros((self.dim,self.dim,self.dim))
        fvol = np.zeros((self.dim,self.dim,self.dim))

        for i in range(self.dim):
            fvol[:,i,:] = sp.ndimage.interpolation.shift(fault,(0,strike[i]),cval=0,prefilter=False,order=1)
            above[:,i,:] = partial_fill_above(fvol[:,i,:])
            below[:,i,:] = partial_fill_below(fvol[:,i,:])

        seisvol = stich_volumes(self.seis,shift_volume_down(self.seis,throw),above,below)
        self.seis = seisvol
        if len(np.unique(self.fault)) == 1:
            self.fault = fvol
        else:
            self.fault = stich_volumes(self.fault,shift_volume_down(self.fault,throw),above,below)
            self.fault += fvol

    
    def __init__(self,dim):
        self.dim=dim
        self.init_seis()
        self.init_fault()


vol = Cube(128)
vol.fold_with_gaussian(10)
vol.single_listric_fault(dip=60,depth_horizontal=2,position=30,throw=10,orientation=40,strike_type="composite")
vol.single_plane_fault(dip=20,position=60,throw=5,orientation=10,strike_type="composite")
vol.single_plane_fault(dip=60,position=70,throw=10,orientation=40,strike_type="composite")

vol.plot_fault_slices(10)
vol.plot_seis_slices(10)

