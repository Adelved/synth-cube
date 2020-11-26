
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.ndimage
import time
import itertools

def normalize(function):
    return (function - function.min() ) / (function.max()- function.min())


def ricker(f, length=1/10, dt=0.001):
    t = np.linspace(-length/2, (length-dt)/2, int(length/dt))
    y = (1.-2.*(np.pi**2)*(f**2)*(t**2))*np.exp(-(np.pi**2)*(f**2)*(t**2))
    return t, y

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

def listric_fault(dim,start_location = 0,depth_horizontal = 1.1, return_values=True):
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
        refl = np.random.normal(vmin,vmax,size=self.dim).repeat(self.dim).reshape(self.dim,self.dim).repeat(self.dim).reshape(self.dim,self.dim,self.dim)
        self.seis = refl

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
    def random_topology(self,num_gaussian,min_smoothing,max_smoothing,max_amplitude):
        topology=0
        for i in range(num_gaussian):
            pos = np.random.randint(self.dim)
            topology+=guassian2D(self.dim,
            pos,
            pos,
            np.random.randint(min_smoothing,max_smoothing),
            np.random.randint(min_smoothing,max_smoothing))
        return normalize(topology) * max_amplitude
    
    def fold_with_gaussian(self,num_gaussian,min_smoothing=30,max_smoothing=100,max_amplitude=5):
        topology = self.random_topology(num_gaussian,min_smoothing,max_smoothing,max_amplitude)
        for iline in range(self.seis.shape[0]):
            for i in range(topology.shape[0]):
                self.seis[:,iline,:][:,i:i+1]=sp.ndimage.interpolation.shift(
                    self.seis[:,iline,:][:,i:i+1],(-topology[:,iline][i],0),cval=0)

    def fold_with_gaussian_fast(self,num_gaussian,min_smoothing=30,max_smoothing=100,max_amplitude=5):
        topology = self.random_topology(num_gaussian,min_smoothing,max_smoothing,max_amplitude)
        for i,j in itertools.product(range(self.dim),range(self.dim)):
            self.seis[:,i,j]=np.roll(self.seis[:,i,j],topology[i,j],axis=0)


        
##Plane fault methods
    def plane_fault_linear_strike(self,dip=45,
    position=50,
    throw=5,
    offset=20,
    mode = None,
    inplace=True):

        x,y = normal_fault(self.dim,position)
        fault = np.zeros((self.dim,self.dim))
        fault[x,y] = 1
        if mode == "random":
            strike = (normalize(np.arange(self.dim))*np.random.randint(0,offset)).astype(int)
        else:
            strike = (normalize(np.arange(self.dim))*offset).astype(int)

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

    def plane_fault_curved_strike(self,dip=45,
    position=50,
    throw=5,
    amplitude=10,
    mode=None,
    inplace=True):

        x,y = normal_fault(self.dim,position)
        fault = np.zeros((self.dim,self.dim))
        fault[x,y] = 1
        if mode == "random":
            strike = (normalize(g(np.arange(self.dim),self.dim//2,self.dim))*np.random.randint(0,amplitude)).astype(int)
        else:
            strike = (normalize(g(np.arange(self.dim),self.dim//2,self.dim))*amplitude).astype(int)


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

    def plane_fault_composite_strike(self,dip=45,
    position=50,
    throw=5,
    offset=10,
    amplitude=10,
    mode = None,
    inplace=True):

        x,y = normal_fault(self.dim,position)
        fault = np.zeros((self.dim,self.dim))
        fault[x,y] = 1
        if mode == "random":
            strike =  ((normalize(g(np.arange(self.dim),self.dim//2,self.dim))*amplitude).astype(int) + 
            (normalize(np.arange(self.dim))*np.random.randint(0,offset)).astype(int) )
        else:
            strike =  ((normalize(g(np.arange(self.dim),self.dim//2,self.dim))*amplitude).astype(int) + 
            (normalize(np.arange(self.dim))*np.random.randint(0,offset)).astype(int) )
            


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

##listric fault methods
    def listric_fault_linear_strike(self,dip=45,
    position=50,
    depth_horizontal=1.1,
    throw=5,
    offset=10,
    mode=None,
    inplace=True):

        x,y = listric_fault(self.dim,position,depth_horizontal)
        fault = np.zeros((self.dim,self.dim))
        fault[x,y] = 1

        if mode == "random":
            strike = (normalize(np.arange(self.dim))*np.random.randint(0,offset)).astype(int)
        else:
            strike = (normalize(np.arange(self.dim))*offset).astype(int)



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

    def listric_fault_curved_strike(self,dip=45,
    position=50,
    depth_horizontal=1.1,
    throw=5,
    amplitude=10,
    mode=None,
    inplace=True):

        x,y = listric_fault(self.dim,position,depth_horizontal)
        fault = np.zeros((self.dim,self.dim))
        fault[x,y] = 1

        if mode == "random":
            strike = (normalize(g(np.arange(self.dim),self.dim//2,self.dim))*np.random.randint(0,amplitude)).astype(int)
        else:
            strike = (normalize(g(np.arange(self.dim),self.dim//2,self.dim))*amplitude).astype(int)



        above = np.zeros((self.dim,self.dim,self.dim))
        below = np.zeros((self.dim,self.dim,self.dim))
        fvol = np.zeros((self.dim,self.dim,self.dim))

        for i in range(self.dim):
            fvol[:,i,:] = sp.ndimage.interpolation.shift(fault,(0,-strike[i]),cval=0,prefilter=False,order=1)
            above[:,i,:] = partial_fill_above(fvol[:,i,:])
            below[:,i,:] = partial_fill_below(fvol[:,i,:])

        seisvol = stich_volumes(self.seis,shift_volume_down(self.seis,throw),above,below)
        self.seis = seisvol
        if len(np.unique(self.fault)) == 1:
            self.fault = fvol
        else:
            self.fault = stich_volumes(self.fault,shift_volume_down(self.fault,throw),above,below)
            self.fault += fvol

    def listric_fault_composite_strike(self,dip=45,
    position=50,
    depth_horizontal=1.1,
    throw=5,
    offset=10,
    amplitude=10,
    mode=None,
    inplace=True):

        x,y = listric_fault(self.dim,position,depth_horizontal)
        fault = np.zeros((self.dim,self.dim))
        fault[x,y] = 1

        if mode == "random":
            strike =  ((normalize(g(np.arange(self.dim),self.dim//2,self.dim))*amplitude).astype(int) + 
            (normalize(np.arange(self.dim))*np.random.randint(0,offset)).astype(int) )
        else:
            strike =  ((normalize(g(np.arange(self.dim),self.dim//2,self.dim))*amplitude).astype(int) + 
            (normalize(np.arange(self.dim))*np.random.randint(0,offset)).astype(int) )


        above = np.zeros((self.dim,self.dim,self.dim))
        below = np.zeros((self.dim,self.dim,self.dim))
        fvol = np.zeros((self.dim,self.dim,self.dim))

        for i in range(self.dim):
            fvol[:,i,:] = sp.ndimage.interpolation.shift(fault,(0,-strike[i]),cval=0,prefilter=False,order=1)
            above[:,i,:] = partial_fill_above(fvol[:,i,:])
            below[:,i,:] = partial_fill_below(fvol[:,i,:])

        seisvol = stich_volumes(self.seis,shift_volume_down(self.seis,throw),above,below)
        self.seis = seisvol
        if len(np.unique(self.fault)) == 1:
            self.fault = fvol
        else:
            self.fault = stich_volumes(self.fault,shift_volume_down(self.fault,throw),above,below)
            self.fault += fvol




##convolution

    def convolve_volume(self,y):
        newvol = np.zeros((self.dim,self.dim,self.dim))
        for iline in range(newvol.shape[1]):
            #temp = sp.ndimage.interpolation.shift(newvol[:,iline,:],(shifts2[iline],0),cval=0)
            newvol[:,iline,:] = np.apply_along_axis(lambda t: np.convolve(t,y,mode='same'),arr=self.seis[:,iline,:],axis=0)
            newvol[:,:,iline] = np.apply_along_axis(lambda t: np.convolve(t,y,mode='same'),arr=self.seis[:,:,iline],axis=0)
            #temp = sp.ndimage.interpolation.shift(newvol[:,:,xline],(shifts[xline],0),cval=0)
        self.seis = newvol
        

    def convolve_noisy_volume(self,y,std=1,fraction=0.5):
        newvol = np.zeros((self.dim,self.dim,self.dim))
        for iline in range(newvol.shape[1]):
            newvol[:,iline,:] = np.apply_along_axis(lambda t: np.convolve(t,y,mode='same'),arr=self.seis[:,iline,:],axis=0)
            temp = sp.ndimage.gaussian_filter(newvol[:,iline,:], std)
            newvol[:,iline,:] = temp + fraction*temp.std() * np.random.random(temp.shape)

            #newvol[:,:,iline] = np.apply_along_axis(lambda t: np.convolve(t,y,mode='same'),arr=self.seis[:,:,iline],axis=0)
            #temp = sp.ndimage.gaussian_filter(newvol[:,:,iline], std)
            #newvol[:,:,iline] = temp + fraction*temp.std() * np.random.random(temp.shape)
        
        self.seis = newvol

    
    
    def __init__(self,dim):
        self.dim=dim
        self.init_seis()
        self.init_fault()

start = time.time()
vol = Cube(256)
end = time.time()
print("time initilize cube",end-start)

start = time.time()
vol.fold_with_gaussian(10,min_smoothing=50,max_smoothing=150,max_amplitude=10)
end = time.time()

vol.plot_seis_slices(150)

print("time folding",end-start)


vol.listric_fault_composite_strike(depth_horizontal=2,offset=30,amplitude=10)

t,y = ricker(50)


#start = time.time()
#vol.convolve_noisy_volume(y,fraction=0.8,std=2)
#end = time.time()
#print("time convolve:",end-start)

vol.plot_fault_slices(150)
vol.plot_seis_slices(150)

