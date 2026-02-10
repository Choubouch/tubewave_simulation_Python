import numpy as np
from numpy import pi as PI

VERBOSE_FIG = 0
FLAG_SKEMPTON = 1


#CONSTRUCTION OF LAYERED MODEL
n = 4
Vpvec = np.array([4000, 4000, 4000, 4000])
Vsvec = np.array([2000, 2000, 2000, 2000])
Rhovec = np.array([2500, 2500, 2500, 2500])
zn_org = np.array([-10, 0, 1])
rn = np.array([0.055, 0.055, 0.065, 0.065])

nu_dyn = 1e-3
Vf = 1500
Rhof = 1000
Kf = Rhof*(Vf**2)

Phivec = 0.3*np.ones(n)
Kappa0_vec = np.zeros(n)

zvec_rec = np.arange(-20.05, 20.05 + 0.1, 0.1)
dt = 0.25e-4
ns = 16001

tvec = np.arange(0, (ns-1)*dt + dt, dt)
dw = 1/tvec[-1]*2*PI
wvec_org = np.arange(0, (ns-1)*dw + dw, dw)

nw_proc = 300
wvec_proc = wvec_org[0:nw_proc]

f0 = 200
delay = 1/(f0**2)

Ricker = 2*PI**2*f0**2*(1.0 - 2.0*(PI**2)*(f0**2)*((tvec-delay)**2))*np.exp(-(PI**2)*(f0**2)*((tvec-delay)**2))
Ricker = Ricker/np.max(np.abs(Ricker))

fRicker = np.conj(np.fft.fft(Ricker))