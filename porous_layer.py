import utils
import numpy as np
from numpy import pi as PI
from scipy.special import kv as besselk
import matplotlib.pyplot as plt

VERBOSE_FIG = 0
FLAG_SKEMPTON = 0

n = 5  # Total number of layers
Vpvec = np.array([5000, 5000, 5000, 5000, 5000], dtype=np.complex128) # size n. P-wave velocity at each layer (m/s)
Vsvec = np.array([3000, 3000, 3000, 3000, 3000], dtype=np.complex128) # size n. S-wave velocity at each layer (m/s)
Rhovec = np.array([2500, 2500, 2500, 2500, 2500], dtype=np.complex128) #Bulk density at each layer (kg/m3)
zn_org = np.array([10, 19.5, 20.5, 50]) # size n-1. The depth of boundaries (m). 
rn = 0.055*np.ones(n) #size n. Borehole radius (m)


nu_dyn = 1e-3 #Fluid dynamic viscosity (Pa*s)
Vf = 1500
Rhof = 1000
Kf = Rhof*Vf**2#Fluid Bulk modulus

# Porous-layer properties 
Phivec = 0.3*np.ones(n) #size n. Porosity (m3/m3)
Kappa0_vec = np.zeros(n, dtype=np.complex128) #size n. Static permeability (m^2), 1 Darcy = 9.869E-13.
Kappa0_vec[2] = 1*9.869e-13 
Vpvec[2] = 4.999654255417096e3
Vsvec[2] = 2.999998101435342e3
Rhovec[2] = 2.5001e3

#TODO implémenter PE hein

zvec_rec = np.arange(-10, 70 + 0.2, 0.2)
dt = 0.25e-4
ns = 16001

tvec = np.arange(0, (ns-1)*dt + dt, dt)
dw = 1/tvec[-1]*2*PI
wvec_org = np.arange(0, (ns-1)*dw + dw, dw)

nw_proc = 300
wvec_proc = wvec_org[0:nw_proc]

f0 = 200
delay = 1/f0*2

fRicker = utils.createRickerWavelet(f0, tvec)

shift_z = 0
zn_proc = zn_org + shift_z

uEvec_allfreq = np.zeros((2, n, nw_proc), dtype=np.complex128)
uEvec_allfreq = utils.createElasticWavefield(n, zn_proc, wvec_proc, Rhovec, Vpvec, shift_z) #ça c'est bon: bah non justemtn
# Necessary elastic moduli and velocities
Evec = Rhovec * Vsvec**2 * (3 * Vpvec**2 - 4 * Vsvec**2) / (Vpvec**2 - Vsvec**2)
mu_vec = Rhovec * Vsvec**2  
lambda_vec = Rhovec * Vpvec**2 - 2 * mu_vec
Kvec = lambda_vec + 2/3 * mu_vec  
CT0vec = np.sqrt(Vf**2 / (1 + Rhof * Vf**2 / (Rhovec * Vsvec**2)))  

# Porous-layer effects: Diffusivity (Ionov, eq. A3)
Diffvec = np.sqrt(Kappa0_vec * Kf / (nu_dyn * Phivec))
TFvec = rn**2 / Diffvec**2  

uvec_allfreq = np.zeros((2, n, nw_proc), dtype=np.complex128)
argsList = (Kf, n, Rhof, rn, uEvec_allfreq, nw_proc, wvec_proc, zn_proc, CT0vec, Evec, Kappa0_vec, Kvec, Phivec, TFvec, Vpvec, Vsvec)
uvec_allfreq = utils.getTubewavePotential(argsList)
argsList = (FLAG_SKEMPTON, nw_proc, fRicker, Rhof, Kf, n, ns, shift_z, zn_proc, Kappa0_vec, Kvec, CT0vec, Evec, Vpvec, Vsvec, uEvec_allfreq, Phivec, TFvec, uvec_allfreq, wvec_proc, zvec_rec)
delay_Haskel = -0.004
data_B = utils.getTimeDomainWaveformBorehole(argsList)

plt.imshow(data_B.T[:, :], vmin=-0.12, vmax=0.12)

plt.show()
