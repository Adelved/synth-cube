
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.ndimage
from fault_utils import *

vol = Cube(128)
vol.fold_with_gaussian(10)
vol.single_normal_fault(dip=20,position=10,throw=5,orientation=40,strike_type="linear")


print(np.unique(vol.fault))
plt.imshow(vol.seis[0,:,:])
