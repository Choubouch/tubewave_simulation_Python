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

shift_z = 0
zn_proc = zn_org + shift_z

uEvec_allfreq = np.zeros((2, n, nw_proc), dtype=np.complex128)

for iw in range(1, nw_proc):
    w = wvec_proc[iw]
    kp = w/Vpvec
    
    Mn = np.zeros((2,2,n-1), dtype=np.complex128)
    for i_n in range(0, n-1):
        kp1 = kp[i_n]
        kp2 = kp[i_n+1]
        z1 = zn_proc[i_n]
        rho1 = Rhovec[i_n]
        rho2 = Rhovec[i_n+1]
        
        a1 = rho2/rho1
        a2 = kp2/kp1 
        
        m11 = 1.0/(2.0)*(a1+a2)*np.exp(1j*(kp1-kp2)*z1) 
        m12 = 1.0/(2.0)*(a1-a2)*np.exp(1j*(kp1+kp2)*z1) 
        m21 = 1.0/(2.0)*(a1-a2)*np.exp(-1j*(kp1+kp2)*z1) 
        m22 = 1.0/(2.0)*(a1+a2)*np.exp(-1j*(kp1-kp2)*z1) 
        
        Mn[:,:,i_n] = np.array([[m11, m12], [m21, m22]], dtype=np.complex128)

    MT = np.eye(2)
    for i_n in range(n-1):
        MT = np.matmul(MT,Mn[:, :, i_n])
    
    uEvec = np.zeros((2,n), dtype=np.complex128)

    D1 = 1/(-Rhovec[1]*w**2)
    
    D1 = D1 * np.exp(1j*kp[1]*(-shift_z))

    Dn = D1/MT[1, 1]
    uEvec[:, n-1] = np.array([0, Dn])

    for i_n in range(n-2, -1, -1):
        uEvec[:,i_n] = Mn[:,:,i_n]@uEvec[:,i_n+1]    

    U1=MT[0,1]/MT[1,1]*D1

    uEvec_allfreq[:,:,iw]=uEvec
    print(uEvec_allfreq[:,:,iw])