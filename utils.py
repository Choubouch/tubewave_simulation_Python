import numpy as np
from scipy.special import k1, kv as besselk
from typing import Tuple, Any
import matplotlib.pyplot as plt
from functools import reduce
import time
from numba import njit

def createRickerWavelet(f0, tvec):
    """
    Parameters :
        f0
        tvec

    Returns :
        fRicker
    """
    delay = 1/f0*2
    Ricker = 2 * np.pi**2 * f0**2 * (1.0 - 2.0 * (np.pi**2) * (f0**2) * ((tvec - delay)**2)) * np.exp(-(np.pi**2) * (f0**2) * ((tvec - delay)**2))
    Ricker = Ricker/np.max(np.abs(Ricker))
    fRicker = np.conj(np.fft.fft(Ricker))
    return fRicker


def getSkemptonCoefficient(kf, phi, vpDry=5170., vsDry=3198.,
                           grainDensity=3143., ks=10e10):
    rhoDry = (1-phi)*grainDensity
    G = vsDry**2*rhoDry
    kDry = vpDry**2*rhoDry - 4/3*G
    B = (1/kDry - 1/ks)/(1/kDry - 1/ks+phi*(1/kf - 1/ks))
    return B


def getElasticWavefield(n, zn_proc, wvec_proc, Rhovec, Vpvec, shift_z):
    """
    Parameters :
        n, kp1, kp2, z1, rho1, rho2
    Returns :
        Mn and MT matrix
    """
    #TODO : enlever les boucles for
    nw_proc = wvec_proc.size
    uEvec_allfreq = np.zeros((2, n, nw_proc), dtype=np.complex64)

    for iw in range(1, nw_proc):
        w = wvec_proc[iw]
        kp = w/Vpvec
        Mn = np.zeros((2,2,n-1), dtype=np.complex64)
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
            Mn[:,:,i_n] = np.array([[m11, m12], [m21, m22]], dtype=np.complex64)

        MT = np.eye(2)
        for i_n in range(n-1):
            MT = np.matmul(MT,Mn[:, :, i_n])
        uEvec = np.zeros((2,n), dtype=np.complex64)

        D1 = 1/(-Rhovec[1]*w**2)
        D1 = D1 * np.exp(1j*kp[1]*(-shift_z))

        Dn = D1/MT[1, 1]
        uEvec[:, n-1] = np.array([0, Dn])


        for i_n in range(n-2, -1, -1):
            uEvec[:,i_n] = Mn[:,:,i_n]@uEvec[:,i_n+1]    

        U1=MT[0,1]/MT[1,1]*D1

        uEvec_allfreq[:,:,iw]=uEvec

    return uEvec_allfreq

# test, amélioration
def getElasticWavefieldOpt(n, zn_proc, wvec_proc, Rhovec, Vpvec, shift_z):
    """
    Parameters :
        n, kp1, kp2, z1, rho1, rho2
    Returns :
        Mn and MT matrix
    """
    #TODO : enlever les boucles for
    nw_proc = wvec_proc.size
    uEvec_allfreq = np.zeros((2, n, nw_proc), dtype=np.complex64)
    w = np.tile(np.reshape(wvec_proc[1:], (-1, 1)), (1, n))
    Vpmat = np.tile(Vpvec, (nw_proc-1, 1))
    kp_mat = w/Vpmat  # kp_mat[i] = le kp de la ième itération de la boucle
    kp1 = kp_mat[:, :-1]
    kp2 = np.roll(kp_mat, -1, axis=1)[:, :-1]
    z1 = zn_proc
    rho1 = Rhovec[:-1]
    rho2 = np.roll(Rhovec, -1)[:-1]
    a1 = rho2/rho1
    a2 = kp2/kp1
    m11 = 1.0/(2.0)*(a1+a2)*np.exp(1j*(kp1-kp2)*z1)
    m12 = 1.0/(2.0)*(a1-a2)*np.exp(1j*(kp1+kp2)*z1)
    m21 = 1.0/(2.0)*(a1-a2)*np.exp(-1j*(kp1+kp2)*z1)
    m22 = 1.0/(2.0)*(a1+a2)*np.exp(-1j*(kp1-kp2)*z1)
    # même Mn, mais Mn[i, :, :, :] = ième Mn de la boucle for
    Mn = np.stack([
        np.stack([m11, m12], axis=1),
        np.stack([m21, m22], axis=1)
        ], axis=1)
    Mn_t = np.moveaxis(Mn, 3, 1)
    MT = np.eye(2)[None, :, :]
    MT = np.repeat(MT, nw_proc-1, axis=0)
    for i in range(n-1):
        MT = MT @ Mn_t[:, i]
    D1 = 1/(-Rhovec[1]*wvec_proc[1:]**2)
    D1 = D1 * np.exp(1j*kp_mat[:, 1]*(-shift_z))
    Dn = D1 / MT[:, 1, 1]
    uEvec_allfreq[:, n-1, 1:] = np.stack([np.zeros_like(Dn), Dn])
    for i_n in range(n-2, -1, -1):
        # j'adore cette ligne
        # faut pas trop poser de questions
        uEvec_allfreq[:,i_n, 1:] = np.moveaxis((Mn[:, :, :, i_n]@np.moveaxis(uEvec_allfreq[ :,i_n+1, 1:], 1, 0)[..., None])[..., 0], 1, 0)

    return uEvec_allfreq


def getFluidResponse(argsList: Tuple[Any, ...]):
    """
    argsList = ( Kf, n, Rhof, rn, uEvec_allfreq, nw_proc, wvec_proc, zn_proc, CT0vec, Evec, Kappa0_vec, Kvec, Phivec, TFvec, Vpvec, Vsvec)
    NB : Cy_square, E1 removed (useless)
    """
    FLAG_SKEMPTON = 1
    Kf, n, Rhof, rn, uEvec_allfreq, nw_proc, wvec_proc, zn_proc, CT0vec, Evec, Kappa0_vec, Kvec, Phivec, TFvec, Vpvec, Vsvec = argsList
    uvec_allfreq = np.zeros((2, n, nw_proc), dtype=np.complex64)
    for iw in range(1, nw_proc):
        w = wvec_proc[iw]
        kp = w/Vpvec

        with np.errstate(divide='ignore', invalid='ignore'):
            tmp = np.sqrt(1j*w*TFvec)
            tmp = tmp.astype(np.complex64)
            PHI_IONOV = (1/tmp) * besselk(1., tmp) / besselk(0., tmp)
            CTvec = CT0vec * np.sqrt(1/(1+2*Phivec*Rhof/Kf*CT0vec**2 * PHI_IONOV))

        CTvec[Kappa0_vec == 0] = CT0vec[Kappa0_vec == 0]
        CTvec = np.conj(CTvec)

        kn = w/CTvec

        Mn = np.zeros((2, 2, n-1), dtype=np.complex64)
        for i_n in range(n-1):
            r1 = rn[i_n]
            r2 = rn[i_n+1]
            k1 = kn[i_n]
            k2 = kn[i_n+1]
            z1 = zn_proc[i_n]
            a1 = r1**2*k1+r2**2*k2
            a2 = r1**2*k1-r2**2*k2
            m11 = a1*np.exp(1j*(k1-k2)*z1)
            m12 = a2*np.exp(1j*(k1+k2)*z1)
            m21 = a2*np.exp(-1j*(k1+k2)*z1)
            m22 = a1*np.exp(-1j*(k1-k2)*z1)

            Mn[:,:,i_n] = 1/(2*r1**2*k1)*np.array([[m11, m12], [m21, m22]])

        Sn = np.zeros((2, n-1), dtype=np.complex64)
        for i_n in range(n-1):
            r1 = rn[i_n]
            r2 = rn[i_n+1]
            k1 = kn[i_n]
            z1 = zn_proc[i_n]
            kp1 = kp[i_n]
            E1 = Evec[i_n]
            Vp_now = Vpvec[i_n]
            Vs_now = Vsvec[i_n]
            CT_now = CTvec[i_n]
            Porosity_now = Phivec[i_n]
            PHI_IONOV_now = np.conj(PHI_IONOV[i_n]) 
            K_now = Kvec[i_n] 

            if i_n == 0 : #gneu gneu les indices sont shiftés de 1
                I1 = 0
                I2 = 0
                I3 = 0
                I4 = 0
            else :
                z0 = zn_proc[i_n-1]
                I1 = np.exp(1j*k1*z1)/1j/(-k1+kp1)*(np.exp(1j*(-k1+kp1)*z1)-np.exp(1j*(-k1+kp1)*z0))-np.exp(-1j*k1*z1)/1j/(k1+kp1)*(np.exp(1j*(k1+kp1)*z1)-np.exp(1j*(k1+kp1)*z0))
                I2 = np.exp(1j*k1*z1)/1j/(-k1-kp1)*(np.exp(1j*(-k1-kp1)*z1)-np.exp(1j*(-k1-kp1)*z0))-np.exp(-1j*k1*z1)/1j/(k1-kp1)*(np.exp(1j*(k1-kp1)*z1)-np.exp(1j*(k1-kp1)*z0))
                I3 = np.exp(1j*k1*z1)/1j/(-k1+kp1)*(np.exp(1j*(-k1+kp1)*z1)-np.exp(1j*(-k1+kp1)*z0))+np.exp(-1j*k1*z1)/1j/(k1+kp1)*(np.exp(1j*(k1+kp1)*z1)-np.exp(1j*(k1+kp1)*z0))
                I4 = np.exp(1j*k1*z1)/1j/(-k1-kp1)*(np.exp(1j*(-k1-kp1)*z1)-np.exp(1j*(-k1-kp1)*z0))+np.exp(-1j*k1*z1)/1j/(k1-kp1)*(np.exp(1j*(k1-kp1)*z1)-np.exp(1j*(k1-kp1)*z0))

            U_Eamp = np.squeeze(uEvec_allfreq[0,i_n,iw]) 
            D_Eamp = np.squeeze(uEvec_allfreq[1,i_n,iw]) 

            A1P = w**2/kp1*(1/(2*Vs_now**2)-1/Vp_now**2)
            delta_p_A = -1j*w*Rhof*CT_now*kp1*A1P*(D_Eamp*I1+U_Eamp*I2)
            delta_vz_A = -1j*w*kp1*A1P*(D_Eamp*I3+U_Eamp*I4)

            if (Kappa0_vec[i_n] !=0):
                delta_p_B = Rhof*CT_now*(-1j*w*Porosity_now/Kf*PHI_IONOV_now)*w**2/Vp_now**2*K_now*(D_Eamp*I1+U_Eamp*I2)
                delta_vz_B = (-1j*w*Porosity_now/Kf*PHI_IONOV_now)*w**2/Vp_now**2*K_now*(D_Eamp*I3+U_Eamp*I4)

                #TODO : traduire le fichier sur skempton 
                if (FLAG_SKEMPTON):
                    B = getSkemptonCoefficient(Kf, 0.3)
                    delta_p_B = B*delta_p_B
                    delta_vz_B = B*delta_vz_B
            else:
                delta_p_B = 0
                delta_vz_B = 0           
            #Une seule occurence de Aninv dans le code : inutile
            #Aninv = [np.exp(i*k1*z1)/(2*w^2*Rhof) -np.exp(i*k1*z1)/(2*w*pi*r1^2*k1) np.exp(-i*k1*z1)/(2*w^2*Rhof) np.exp(-i*k1*z1)/(2*w*pi*r1^2*k1)] 
            tmpdata = -1j*kp1*U_Eamp.T * np.exp(-1j*kp1*z1)+1j*kp1*D_Eamp.T*np.exp(1j*kp1*z1) 
            tmpdata = -1j*w*tmpdata 
            dVn = np.pi*(r2**2-r1**2)*tmpdata  
            dvz = dVn/(np.pi*r1**2) 
                            
            dp_total = delta_p_A+delta_p_B
            dvz_total = delta_vz_A+delta_vz_B+dvz
            
            
            
            Sn[:, i_n] = 1/(2*Rhof*w**2*k1)*np.array([(Rhof*w*dvz_total-k1*dp_total)*np.exp(1j*k1*z1), -(Rhof*w*dvz_total+k1*dp_total)*np.exp(-1j*k1*z1)])
            

        MT = np.eye(2)
        ST = np.zeros(2)
        for i_n in range(n-1):
            ST = ST+MT@Sn[:,i_n]
            MT = MT@Mn[:,:,i_n]
        uvec = np.zeros((2,n), dtype=np.complex64)
        D_Eamp = np.squeeze(uEvec_allfreq[1,n-1,iw]) 
        z0 = zn_proc[n-2]
        k1 = kn[n-1]
        kp1 = kp[n-1]
        Vp_now = Vpvec[n-1]
        Vs_now = Vsvec[n-1]
        CT_now = CTvec[n-1]
        I1 = 2*k1/1j/(kp1**2-k1**2)*(np.exp(1j*kp1*z0))
        I3 = 2*kp1/1j/(kp1**2-k1**2)*(np.exp(1j*kp1*z0))


        A1P = w**2/kp1*(1/(2*Vs_now**2)-1/Vp_now**2)
        delta_p_A = -1j*w*Rhof*CT_now*kp1*A1P*(D_Eamp*I1)
        delta_vz_A = -1j*w*kp1*A1P*(D_Eamp*I3)
        
        Un = np.exp(1j*k1*z0)/(2*Rhof*w**2*k1)*(k1*delta_p_A-Rhof*w*delta_vz_A)

        D_Eamp = np.squeeze(uEvec_allfreq[1,0,iw]) 
        U_Eamp = np.squeeze(uEvec_allfreq[0,0,iw]) 
        
        
        z1 = zn_proc[0]
        k1 = kn[0]
        kp1 = kp[0]
        Vp_now = Vpvec[0]
        Vs_now = Vsvec[0]
        CT_now = CTvec[0]

        
        I1 = 2*k1/1j/(kp1**2-k1**2)*(np.exp(1j*kp1*z1))
        I2 = 2*k1/1j/(kp1**2-k1**2)*(np.exp(-1j*kp1*z1))
        I3 = 2*kp1/1j/(kp1**2-k1**2)*(np.exp(1j*kp1*z1))
        I4 = -2*kp1/1j/(kp1**2-k1**2)*(np.exp(-1j*kp1*z1))

        A1P = w**2/kp1*(1/(2*Vs_now**2)-1/Vp_now**2)
        delta_p_A = -1j*w*Rhof*CT_now*kp1*A1P*(D_Eamp*I1+U_Eamp*I2)
        delta_vz_A = -1j*w*kp1*A1P*(D_Eamp*I3+U_Eamp*I4)
        
        
        D1 = np.exp(-1j*k1*z1)/(2*Rhof*w**2*k1)*(k1*delta_p_A+Rhof*w*delta_vz_A)
        Dn = (D1-MT[1,0]*Un-ST[1])/MT[1, 1]
        U1 = MT[0, 0]*Un+MT[0, 1]/MT[1, 1]*(D1-MT[1, 0]*Un-ST[1])+ST[0]
        uvec[:,n-1] = np.array([Un, Dn])

        for i_n in range(n-2, -1, -1):
            uvec[:,i_n] = Mn[:,:,i_n]@uvec[:,i_n+1]+Sn[:,i_n]      
        uvec_allfreq[:,:,iw] = uvec


    return uvec_allfreq

def getFluidResponseOpt(argsList: Tuple[Any, ...]):
    """
    argsList = ( Kf, n, Rhof, rn, uEvec_allfreq, nw_proc, wvec_proc, zn_proc, CT0vec, Evec, Kappa0_vec, Kvec, Phivec, TFvec, Vpvec, Vsvec)
    NB : Cy_square, E1 removed (useless)
    """
    FLAG_SKEMPTON = 1
    Kf, n, Rhof, rn, uEvec_allfreq, nw_proc, wvec_proc, zn_proc, CT0vec, Evec, Kappa0_vec, Kvec, Phivec, TFvec, Vpvec, Vsvec = argsList
    uvec_allfreq = np.zeros((2, n, nw_proc), dtype=np.complex64)
    for iw in range(1, nw_proc):
        w = wvec_proc[iw]
        kp = w/Vpvec

        with np.errstate(divide='ignore', invalid='ignore'):
            tmp = np.sqrt(1j*w*TFvec)
            tmp = tmp.astype(np.complex64)
            PHI_IONOV = (1/tmp) * besselk(1., tmp) / besselk(0., tmp)
            CTvec = CT0vec * np.sqrt(1/(1+2*Phivec*Rhof/Kf*CT0vec**2 * PHI_IONOV))

        CTvec[Kappa0_vec == 0] = CT0vec[Kappa0_vec == 0]
        CTvec = np.conj(CTvec)

        kn = w/CTvec
        # kp1 = kp[:-1]
        # kp2 = np.roll(kp, -1)[:-1]
        # z1 = zn_proc
        # rho1 = Rhovec[:-1]
        # rho2 = np.roll(Rhovec, -1)[-1]
        # a1 = rho2/rho1
        # a2 = kp2/kp1
        # m11 = 1.0/(2.0)*(a1+a2)*np.exp(1j*(kp1-kp2)*z1)
        # m12 = 1.0/(2.0)*(a1-a2)*np.exp(1j*(kp1+kp2)*z1)
        # m21 = 1.0/(2.0)*(a1-a2)*np.exp(-1j*(kp1+kp2)*z1)
        # m22 = 1.0/(2.0)*(a1+a2)*np.exp(-1j*(kp1-kp2)*z1)
        # # Mn[:,:,] = np.array([[m11, m12], [m21, m22]], dtype=np.complex64)
        # Mn = np.array([[m11.T, m12.T], [m21.T, m22.T]], dtype=np.complex64)
        Mn = np.zeros((2, 2, n-1), dtype=np.complex64)
        z1 = zn_proc
        r1 = rn[:-1]
        r2 = np.roll(rn, -1)[: -1]
        k1 = kn[:-1]
        k2 = np.roll(kn, -1)[: -1]
        a1 = r1**2*k1+r2**2*k2
        a2 = r1**2*k1-r2**2*k2
        m11 = a1*np.exp(1j*(k1-k2)*z1)
        m12 = a2*np.exp(1j*(k1+k2)*z1)
        m21 = a2*np.exp(-1j*(k1+k2)*z1)
        m22 = a1*np.exp(-1j*(k1-k2)*z1)
        Mn = np.array([[m11.T, m12.T], [m21.T, m22.T]], dtype=np.complex64)
        Mn = 1/(2*r1**2*k1)*Mn
        # print(m11)
        # for i_n in range(n-1):
        #     r1 = rn[i_n]
        #     r2 = rn[i_n+1]
        #     k1 = kn[i_n]
        #     k2 = kn[i_n+1]
        #     z1 = zn_proc[i_n]
        #     a1 = r1**2*k1+r2**2*k2
        #     a2 = r1**2*k1-r2**2*k2
        #     m11 = a1*np.exp(1j*(k1-k2)*z1)
        #     m12 = a2*np.exp(1j*(k1+k2)*z1)
        #     m21 = a2*np.exp(-1j*(k1+k2)*z1)
        #     m22 = a1*np.exp(-1j*(k1-k2)*z1)
        #     print(m11)
        #     input()
        #     Mn[:,:,i_n] = 1/(2*r1**2*k1)*np.array([[m11, m12], [m21, m22]])

        Sn = np.zeros((2, n-1), dtype=np.complex64)
        for i_n in range(n-1):
            r1 = rn[i_n]
            r2 = rn[i_n+1]
            k1 = kn[i_n]
            z1 = zn_proc[i_n]
            kp1 = kp[i_n]
            E1 = Evec[i_n]
            Vp_now = Vpvec[i_n]
            Vs_now = Vsvec[i_n]
            CT_now = CTvec[i_n]
            Porosity_now = Phivec[i_n]
            PHI_IONOV_now = np.conj(PHI_IONOV[i_n]) 
            K_now = Kvec[i_n] 

            if i_n == 0 : #gneu gneu les indices sont shiftés de 1
                I1 = 0
                I2 = 0
                I3 = 0
                I4 = 0
            else :
                z0 = zn_proc[i_n-1]
                I1 = np.exp(1j*k1*z1)/1j/(-k1+kp1)*(np.exp(1j*(-k1+kp1)*z1)-np.exp(1j*(-k1+kp1)*z0))-np.exp(-1j*k1*z1)/1j/(k1+kp1)*(np.exp(1j*(k1+kp1)*z1)-np.exp(1j*(k1+kp1)*z0))
                I2 = np.exp(1j*k1*z1)/1j/(-k1-kp1)*(np.exp(1j*(-k1-kp1)*z1)-np.exp(1j*(-k1-kp1)*z0))-np.exp(-1j*k1*z1)/1j/(k1-kp1)*(np.exp(1j*(k1-kp1)*z1)-np.exp(1j*(k1-kp1)*z0))
                I3 = np.exp(1j*k1*z1)/1j/(-k1+kp1)*(np.exp(1j*(-k1+kp1)*z1)-np.exp(1j*(-k1+kp1)*z0))+np.exp(-1j*k1*z1)/1j/(k1+kp1)*(np.exp(1j*(k1+kp1)*z1)-np.exp(1j*(k1+kp1)*z0))
                I4 = np.exp(1j*k1*z1)/1j/(-k1-kp1)*(np.exp(1j*(-k1-kp1)*z1)-np.exp(1j*(-k1-kp1)*z0))+np.exp(-1j*k1*z1)/1j/(k1-kp1)*(np.exp(1j*(k1-kp1)*z1)-np.exp(1j*(k1-kp1)*z0))

            U_Eamp = np.squeeze(uEvec_allfreq[0,i_n,iw]) 
            D_Eamp = np.squeeze(uEvec_allfreq[1,i_n,iw]) 

            A1P = w**2/kp1*(1/(2*Vs_now**2)-1/Vp_now**2)
            delta_p_A = -1j*w*Rhof*CT_now*kp1*A1P*(D_Eamp*I1+U_Eamp*I2)
            delta_vz_A = -1j*w*kp1*A1P*(D_Eamp*I3+U_Eamp*I4)

            if (Kappa0_vec[i_n] !=0):
                delta_p_B = Rhof*CT_now*(-1j*w*Porosity_now/Kf*PHI_IONOV_now)*w**2/Vp_now**2*K_now*(D_Eamp*I1+U_Eamp*I2)
                delta_vz_B = (-1j*w*Porosity_now/Kf*PHI_IONOV_now)*w**2/Vp_now**2*K_now*(D_Eamp*I3+U_Eamp*I4)

                #TODO : traduire le fichier sur skempton 
                if (FLAG_SKEMPTON):
                    B = getSkemptonCoefficient(Kf, 0.3)
                    delta_p_B = B*delta_p_B
                    delta_vz_B = B*delta_vz_B
            else:
                delta_p_B = 0
                delta_vz_B = 0           
            #Une seule occurence de Aninv dans le code : inutile
            #Aninv = [np.exp(i*k1*z1)/(2*w^2*Rhof) -np.exp(i*k1*z1)/(2*w*pi*r1^2*k1) np.exp(-i*k1*z1)/(2*w^2*Rhof) np.exp(-i*k1*z1)/(2*w*pi*r1^2*k1)] 
            tmpdata = -1j*kp1*U_Eamp.T * np.exp(-1j*kp1*z1)+1j*kp1*D_Eamp.T*np.exp(1j*kp1*z1) 
            tmpdata = -1j*w*tmpdata 
            dVn = np.pi*(r2**2-r1**2)*tmpdata  
            dvz = dVn/(np.pi*r1**2) 
                            
            dp_total = delta_p_A+delta_p_B
            dvz_total = delta_vz_A+delta_vz_B+dvz
            
            
            
            Sn[:, i_n] = 1/(2*Rhof*w**2*k1)*np.array([(Rhof*w*dvz_total-k1*dp_total)*np.exp(1j*k1*z1), -(Rhof*w*dvz_total+k1*dp_total)*np.exp(-1j*k1*z1)])
            

        MT = np.eye(2)
        ST = np.zeros(2)
        for i_n in range(n-1):
            ST = ST+MT@Sn[:,i_n]
            MT = MT@Mn[:,:,i_n]
        uvec = np.zeros((2,n), dtype=np.complex64)
        D_Eamp = np.squeeze(uEvec_allfreq[1,n-1,iw]) 
        z0 = zn_proc[n-2]
        k1 = kn[n-1]
        kp1 = kp[n-1]
        Vp_now = Vpvec[n-1]
        Vs_now = Vsvec[n-1]
        CT_now = CTvec[n-1]
        I1 = 2*k1/1j/(kp1**2-k1**2)*(np.exp(1j*kp1*z0))
        I3 = 2*kp1/1j/(kp1**2-k1**2)*(np.exp(1j*kp1*z0))


        A1P = w**2/kp1*(1/(2*Vs_now**2)-1/Vp_now**2)
        delta_p_A = -1j*w*Rhof*CT_now*kp1*A1P*(D_Eamp*I1)
        delta_vz_A = -1j*w*kp1*A1P*(D_Eamp*I3)
        
        Un = np.exp(1j*k1*z0)/(2*Rhof*w**2*k1)*(k1*delta_p_A-Rhof*w*delta_vz_A)

        D_Eamp = np.squeeze(uEvec_allfreq[1,0,iw]) 
        U_Eamp = np.squeeze(uEvec_allfreq[0,0,iw]) 
        
        
        z1 = zn_proc[0]
        k1 = kn[0]
        kp1 = kp[0]
        Vp_now = Vpvec[0]
        Vs_now = Vsvec[0]
        CT_now = CTvec[0]

        
        I1 = 2*k1/1j/(kp1**2-k1**2)*(np.exp(1j*kp1*z1))
        I2 = 2*k1/1j/(kp1**2-k1**2)*(np.exp(-1j*kp1*z1))
        I3 = 2*kp1/1j/(kp1**2-k1**2)*(np.exp(1j*kp1*z1))
        I4 = -2*kp1/1j/(kp1**2-k1**2)*(np.exp(-1j*kp1*z1))

        A1P = w**2/kp1*(1/(2*Vs_now**2)-1/Vp_now**2)
        delta_p_A = -1j*w*Rhof*CT_now*kp1*A1P*(D_Eamp*I1+U_Eamp*I2)
        delta_vz_A = -1j*w*kp1*A1P*(D_Eamp*I3+U_Eamp*I4)
        
        
        D1 = np.exp(-1j*k1*z1)/(2*Rhof*w**2*k1)*(k1*delta_p_A+Rhof*w*delta_vz_A)
        Dn = (D1-MT[1,0]*Un-ST[1])/MT[1, 1]
        U1 = MT[0, 0]*Un+MT[0, 1]/MT[1, 1]*(D1-MT[1, 0]*Un-ST[1])+ST[0]
        uvec[:,n-1] = np.array([Un, Dn])

        for i_n in range(n-2, -1, -1):
            uvec[:,i_n] = Mn[:,:,i_n]@uvec[:,i_n+1]+Sn[:,i_n]      
        uvec_allfreq[:,:,iw] = uvec


    return uvec_allfreq


def getTimeDomainWaveformBorehole(argsList):
    """
    argList : (FLAG_SKEMPTON, nw_proc, fRicker, Rhof, Kf, n, ns, shift_z, zn_proc, Kappa0_vec, Kvec, CT0vec, Evec, Vpvec, Vsvec, uEvec_allfreq, Phivec, TFvec, uvec_allfreq, wvec_proc, zvec_rec)
    """
    FLAG_SKEMPTON, nw_proc, fRicker, Rhof, Kf, n, ns, shift_z, zn_proc, Kappa0_vec, Kvec, CT0vec, Evec, Vpvec, Vsvec, uEvec_allfreq, Phivec, TFvec, uvec_allfreq, wvec_proc, zvec_rec= argsList
    fdata = np.zeros((zvec_rec.size , ns), dtype=np.complex64)
    tempsInt = 0
    for irec in range(zvec_rec.size) :
        znow = zvec_rec[irec]+shift_z
        #Detecting a layer number
        tmp = zn_proc-znow
        tmp[tmp<0] = np.inf
        tmp1 = np.min(tmp)
        tmp2 = np.argmin(tmp)
        in_now = tmp2
        if(sum(tmp==np.inf) == tmp.size):
            in_now = (zn_proc.size) #last layer

        U_amp = np.squeeze(uvec_allfreq[0,in_now,:]) #upgoing,layer in_now
        D_amp = np.squeeze(uvec_allfreq[1,in_now,:]) #downgoing,layer in_now


        if(Kappa0_vec[in_now] != 0):
            #Porous layer effects: Calculating the function PHI of Ionov
            #   and correponding tube-wave velocities
            #Note: all variables related to the porous layer assumes MATLAB-FT
            tmp = np.sqrt(1j*wvec_proc*TFvec[in_now]) #be carefull INF when kappa0 = 0
            tmp = tmp.astype(np.complex64)
            PHI_IONOV = tmp**(-1)*besselk(1,tmp)/besselk(0,tmp) #be careful NaN when kappa0 = 0

            #Tube-wave velocity including the porous-layer effects
            CTvec = CT0vec[in_now]*np.sqrt(1/(1+2*Phivec[in_now]*Rhof/Kf*CT0vec[in_now]**2*PHI_IONOV)) #this is a frequency-vector version.

            CTvec = np.conj(CTvec) #MATLAB FT -> Aki-Richards FT
            kcvec = wvec_proc/CTvec
        else :
            kcvec = wvec_proc/CT0vec[in_now]

        phi = U_amp.T*np.exp(-1j*kcvec*znow)+D_amp.T*np.exp(1j*kcvec*znow) #displacement potential (note: kc homogenous)
        tmpdata = Rhof*wvec_proc**2*phi #pressure
        #Evalauting the discontinuities (delta_p and delta_vz)
        #due to external effective stress.
        #Note:multiple-freq vector version    
        #I am at z1 = zvec(irec). z0 is the nearest boundary in the negative z direction
        if(in_now == 0):
            z1 = znow 
            z0 = zn_proc[0]
        elif(in_now == n-1):
            z1 = znow 
            z0 = zn_proc[n-2]
        else :
            z1 = znow 
            z0 = zn_proc[in_now-1]
        #I am at in = in_now
        k1vec = kcvec    
        kp1vec = wvec_proc/Vpvec[in_now]
        E1 = Evec[in_now]
        Vp_now = Vpvec[in_now]
        Vs_now = Vsvec[in_now]
        vieuxTemps = time.time()
        with np.errstate(divide='ignore', invalid='ignore'):
        #Integration element I1:
            I1 = np.exp(1j*k1vec*z1)/1j/(-k1vec+kp1vec)*(np.exp(1j*(-k1vec+kp1vec)*z1)-np.exp(1j*(-k1vec+kp1vec)*z0))-np.exp(-1j*k1vec*z1)/1j/(k1vec+kp1vec)*(np.exp(1j*(k1vec+kp1vec)*z1)-np.exp(1j*(k1vec+kp1vec)*z0))
        #Integration element I2:
            I2 = np.exp(1j*k1vec*z1)/1j/(-k1vec-kp1vec)*(np.exp(1j*(-k1vec-kp1vec)*z1)-np.exp(1j*(-k1vec-kp1vec)*z0))-np.exp(-1j*k1vec*z1)/1j/(k1vec-kp1vec)*(np.exp(1j*(k1vec-kp1vec)*z1)-np.exp(1j*(k1vec-kp1vec)*z0))
        #Integration element I3:
            # I3 = np.exp(1j*k1vec*z1)/1j/(-k1vec+kp1vec)*(np.exp(1j*(-k1vec+kp1vec)*z1)-np.exp(1j*(-k1vec+kp1vec)*z0))+np.exp(-1j*k1vec*z1)/1j/(k1vec+kp1vec)*(np.exp(1j*(k1vec+kp1vec)*z1)-np.exp(1j*(k1vec+kp1vec)*z0))
        #Integration element I4:
            # I4 = np.exp(1j*k1vec*z1)/1j/(-k1vec-kp1vec)*(np.exp(1j*(-k1vec-kp1vec)*z1)-np.exp(1j*(-k1vec-kp1vec)*z0))+np.exp(-1j*k1vec*z1)/1j/(k1vec-kp1vec)*(np.exp(1j*(k1vec-kp1vec)*z1)-np.exp(1j*(k1vec-kp1vec)*z0))
        tempsInt += time.time() - vieuxTemps
        if(Kappa0_vec[in_now] != 0):
            CT_now = CTvec #vector
        else:
            CT_now = CT0vec[in_now] #scalar

        #---the discontinuities (delta_p and delta_vz) due to elastic wave---
        U_Eamp = np.squeeze(uEvec_allfreq[0,in_now,:]) #upgoing elastic wave,layer in_now
        D_Eamp = np.squeeze(uEvec_allfreq[1,in_now,:]) #downgoing elastic wave,layer in_now

        U_Eamp = U_Eamp.T
        D_Eamp = D_Eamp.T

        with np.errstate(divide='ignore', invalid='ignore'):
            A1P = wvec_proc**2/kp1vec*(1/(2*Vs_now**2)-1/Vp_now**2)
        delta_p_A = -1j*wvec_proc*Rhof*CT_now*kp1vec*A1P*(D_Eamp*I1+U_Eamp*I2)
        #the discontinuities (delta_p and delta_vz) due to a porous layer 
        #(PHI_IONOV_now from Aki-Richards FT)
        if(Kappa0_vec[in_now] != 0):
            Porosity_now = Phivec[in_now]
            K_now = Kvec[in_now]

            delta_p_B = Rhof*CT_now*(-1j*wvec_proc*Porosity_now/Kf*np.conj(PHI_IONOV))*wvec_proc**2/Vp_now**2*K_now*(D_Eamp*I1+U_Eamp*I2) #BUGFIX
            ##Option B
            if (FLAG_SKEMPTON):
                    # The following two lines calculates the Skempton coefficient (B)
                    # assuming the given Vp and Vs to be those in the drained condition.
                    # This is not recommed.
    # $$$                 K_d = K_now #assuming Vp and Vs are drained
    # $$$                 B = 1-1/(1+Kf/(Porosity_now*K_d)) #skepmton coefficient
                    # The following line uses the Skempton coefficient (B) 
                    # provided in PE.B (recommed)
                    #TODO : comme le todo d'avant là
                B = getSkemptonCoefficient(Kf, 0.3)
                    #Application of the Skempton coefficient (B)
                delta_p_B = B*delta_p_B
        else:
            delta_p_B = 0
        tmpdata = tmpdata+delta_p_A+delta_p_B
        tmpdata = tmpdata*fRicker[0:nw_proc] #Note: "Unit incident wave" is
                                            # defined in elastic-wave potentials
        fdata[irec,0:nw_proc] = tmpdata

    fdata[:,0] = 0
    fdata = np.conj(fdata); #Aki-Richards FT -> MATLAB FT
    fdata[:, 1:] = fdata[:, 1:] + np.conj(np.flip(fdata[:, 1:], axis=1)) #tour de passe-passe pour avoir la symétrie des données 
    data = (np.fft.ifft(fdata, axis=1)).real
    data_B = data.T
    # print("int")
    # print(tempsInt)
    return data_B

@njit
def fastI(k1vec, kp1vec, z0, z1):
    I1 = np.zeros_like(k1vec, dtype=np.complex128)
    I2 = np.zeros_like(k1vec, dtype=np.complex128)
    for i in range(k1vec.shape[0]):
        k1 = k1vec[i][1:-1]
        kp1 = kp1vec[i][1:-1]
        z0i = z0[i][1:-1]
        z1i = z1[i][1:-1]
        den1 = (-k1+kp1)
        den2 = (k1+kp1)
        den3 = (-k1-kp1)
        den4 = (k1-kp1)
        #if (den1 == 0 or den2 == 0):
        #    I1[i] = np.inf
        #else :
        I1[i][1:-1] = np.exp(1j*k1*z1i)/1j/den1*(np.exp(1j*(-k1+kp1)*z1i)-np.exp(1j*(-k1+kp1)*z0i))-np.exp(-1j*k1*z1i)/1j/den2*(np.exp(1j*(k1+kp1)*z1i)-np.exp(1j*(k1+kp1)*z0i))
        #if (den3 == 0 or den4 == 0):
        #    I2[i] == np.inf
        #else : 
        I2[i][1:-1] = np.exp(1j*k1*z1i)/1j/den3*(np.exp(1j*(-k1-kp1)*z1i)-np.exp(1j*(-k1-kp1)*z0i))-np.exp(-1j*k1*z1i)/1j/den4*(np.exp(1j*(k1-kp1)*z1i)-np.exp(1j*(k1-kp1)*z0i))
    return I1, I2

def getTimeDomainWaveformBoreholeOpt(argsList):
    """
    argList : (FLAG_SKEMPTON, nw_proc, fRicker, Rhof, Kf, n, ns, shift_z, zn_proc, Kappa0_vec, Kvec, CT0vec, Evec, Vpvec, Vsvec, uEvec_allfreq, Phivec, TFvec, uvec_allfreq, wvec_proc, zvec_rec)
    """
    FLAG_SKEMPTON, nw_proc, fRicker, Rhof, Kf, n, ns, shift_z, zn_proc, Kappa0_vec, Kvec, CT0vec, Evec, Vpvec, Vsvec, uEvec_allfreq, Phivec, TFvec, uvec_allfreq, wvec_proc, zvec_rec= argsList
    fdata = np.zeros((zvec_rec.size , ns), dtype=np.complex64)
    znow = np.tile((zvec_rec + shift_z), (zn_proc.size, 1))
    znow = np.moveaxis(znow, 1, 0)
    tmpMat = np.tile(zn_proc, (zvec_rec.size, 1)) - znow
    tmpMat[tmpMat < 0] = np.inf
    tmp2 = np.argmin(tmpMat, axis=1)
    in_now = tmp2
    mask = np.all(tmpMat == np.inf, axis=1)
    in_now[mask] = zn_proc.size
    U_amp = uvec_allfreq[0, in_now, :]
    D_amp = uvec_allfreq[1, in_now, :]
    mask = (Kappa0_vec[in_now] != 0)
    kcvec = np.empty((zvec_rec.shape[0],
                      wvec_proc.shape[0]), dtype=np.complex64)
    # in_now -> (13001, ) indice pour chaque itération
    # wvec_proc -> (1200, )
    # CT0vec -> (56, )
# kcvec -> (13001, 1200)
    wvec_procMat = np.tile(wvec_proc, (zvec_rec.shape[0], 1))
    kcvec[~mask] = wvec_procMat[~mask]/np.moveaxis(np.tile(CT0vec[in_now],
                                                       (wvec_proc.shape[0],1)),
                                                       0, 1)[~mask]
    if np.any(mask):
        tmpMat = np.moveaxis(np.tile(TFvec[in_now][mask],
                                     (wvec_proc.shape[0],1)), 0, 1)

        tmp = np.sqrt(1j*wvec_procMat[mask]*tmpMat)

        PHI_IONOV = tmp**(-1)*besselk(1,tmp)/besselk(0,tmp) #be careful NaN when kappa0 = 0
            #Tube-wave velocity including the porous-layer effects
        tmpMat = np.moveaxis(np.tile(CT0vec[in_now][mask],
                                     (wvec_proc.shape[0],1)), 0, 1)
        tmpMat2 = np.moveaxis(np.tile(Phivec[in_now][mask],
                                     (wvec_proc.shape[0],1)), 0, 1)
        CTvec = tmpMat*np.sqrt(1/(1+2*tmpMat2*Rhof/Kf*tmpMat**2*PHI_IONOV)) #this is a frequency-vector version.
        CTvec = np.conj(CTvec) #MATLAB FT -> Aki-Richards FT
        kcvec[mask] = wvec_procMat[mask]/CTvec
    znow = np.tile((zvec_rec + shift_z), (kcvec.shape[1], 1))
    znow = np.moveaxis(znow, 1, 0)
    phi = U_amp*np.exp(-1j*kcvec*znow)+D_amp*np.exp(1j*kcvec*znow) #displacement potential (note: kc homogenous)
    tmpdata = Rhof*wvec_proc**2*phi #pressure
    znow = zvec_rec + shift_z
    z1 = np.zeros_like(in_now, dtype=np.float32)
    z0 = np.zeros_like(in_now, dtype=np.float32)
    mask = (in_now == 0)
    z1[mask] = znow[mask]
    z0[mask] = zn_proc[0]
    mask = (in_now == n-1)
    z1[mask] = znow[mask]
    z0[mask] = zn_proc[n-2]
    mask = (in_now != 0)
    mask[in_now == n-2] = False
    z1[mask] = znow[mask]
    z0[mask] = zn_proc[in_now[mask] - 1]

    k1vec = kcvec
    VpvecMat = np.tile(Vpvec[in_now], (wvec_proc.shape[0], 1))
    VpvecMat = np.moveaxis(VpvecMat, 0, 1)
    kp1vec = wvec_proc/VpvecMat
    Vp_now = Vpvec[in_now]
    Vs_now = Vsvec[in_now]
    print(z1.shape)
    # z1 = np.tile(z1, (wvec_proc.shape[0], 1))
    # z1 = np.moveaxis(z1, 0, 1)
    # z0 = np.tile(z0, (wvec_proc.shape[0], 1))
    # z0 = np.moveaxis(z0, 0, 1)
    z1 = z1[:, np.newaxis]
    z0 = z0[:, np.newaxis]
    #print("int")
    #vieuxTemps = time.time()
    with np.errstate(divide='ignore', invalid='ignore'):
#Integration element I1:
        A = -k1vec + kp1vec
        B =  k1vec + kp1vec
        C = -B # -k1vec - kp1vec
        D = -A #k1vec - kp1vec
        exp1 = np.exp(1j * k1vec * z1)
        exp2 = np.exp(-1j * k1vec * z1)

        expA_z1 = np.exp(1j * A * z1)
        expA_z0 = np.exp(1j * A * z0)
       #
        expB_z1 = np.exp(1j * B * z1)
        expB_z0 = np.exp(1j * B * z0)
#
        expC_z1 = np.exp(1j * C * z1)
        expC_z0 = np.exp(1j * C * z0)
#
        expD_z1 = np.exp(1j * C * z1)
        expD_z0 = np.exp(1j * C * z0)
        term1 = exp1 / (1j * A) * (expA_z1 - expA_z0)
        term2 = exp2 / (1j * B) * (expB_z1 - expB_z0)

        term3 = exp1 / (1j * C) * (expC_z1 - expC_z0)
        term4 = exp2 / (1j * D) * (expD_z1 - expD_z0)
        I1 = term1 - term2
        #Integration element I2
        I2 = term3 - term4
        #I1, I2 = fastI(k1vec, kp1vec, z0, z1)
               #Integration element I3:
        #I3 = np.exp(1j*k1vec*z1)/1j/(-k1vec+kp1vec)*(np.exp(1j*(-k1vec+kp1vec)*z1)-np.exp(1j*(-k1vec+kp1vec)*z0))+np.exp(-1j*k1vec*z1)/1j/(k1vec+kp1vec)*(np.exp(1j*(k1vec+kp1vec)*z1)-np.exp(1j*(k1vec+kp1vec)*z0))
       #Integration element I4:
    #I4 = np.exp(1j*k1vec*z1)/1j/(-k1vec-kp1vec)*(np.exp(1j*(-k1vec-kp1vec)*z1)-np.exp(1j*(-k1vec-kp1vec)*z0))+np.exp(-1j*k1vec*z1)/1j/(k1vec-kp1vec)*(np.exp(1j*(k1vec-kp1vec)*z1)-np.exp(1j*(k1vec-kp1vec)*z0))
    print(time.time() - vieuxTemps)
    mask = (Kappa0_vec[in_now] != 0)
    CT_now = np.empty((in_now.shape[0], wvec_proc.shape[0]))
    mask = (Kappa0_vec[in_now] != 0)
    if np.any(mask):
        CT_now[mask] = CTvec
    CT_now[~mask, :] = np.moveaxis(np.tile(CT0vec[in_now][~mask], (wvec_proc.shape[0], 1)), 0, 1)
    U_Eamp = uEvec_allfreq[0, in_now, :]
    D_Eamp = uEvec_allfreq[1, in_now, :]
    Vs_now = np.tile(Vs_now, (wvec_proc.shape[0], 1))
    Vs_now = np.moveaxis(Vs_now, 0, 1)
    Vp_now = np.tile(Vp_now, (wvec_proc.shape[0], 1))
    Vp_now = np.moveaxis(Vp_now, 0, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        A1P = wvec_proc**2/kp1vec*(1/(2*Vs_now**2)-1/Vp_now**2)
    delta_p_A = -1j*wvec_proc*Rhof*CT_now*kp1vec*A1P*(D_Eamp*I1+U_Eamp*I2)
    tmpdata = tmpdata+delta_p_A
#    fRickerMat = np.tile(fRicker[0:nw_proc], (in_now.shape[0], 1))
    tmpdata = tmpdata*fRicker[0:nw_proc]
    fdata[:,0:nw_proc] = tmpdata
    fdata[:,0] = 0
    fdata = np.conj(fdata); #Aki-Richards FT -> MATLAB FT
    fdata[:, 1:] = fdata[:, 1:] + np.conj(np.flip(fdata[:, 1:], axis=1)) #tour de passe-passe pour avoir la symétrie des données 
    data = (np.fft.ifft(fdata, axis=1)).real
    data_B = data.T
    
    return data_B
    # Salut moi du turfu : jusqu'ici tout va bien
    # kcvecjaaj[i] c'est bien le i eme kcvec
    # attention aux conversions en cx64 qui peuvent faire des minis différences
    # + numpy est con et dit que nan est différent de nan 
    # sinon j'espère qu'il ne faudra pas revenir sur le code
    # car c'est incompréhensible
    # à moins de rentrer en trans en se droguant très fort

def getTimeDomainWaveformBoreholeOptV2(argsList):
    """
    argList : (FLAG_SKEMPTON, nw_proc, fRicker, Rhof, Kf, n, ns, shift_z, zn_proc, Kappa0_vec, Kvec, CT0vec, Evec, Vpvec, Vsvec, uEvec_allfreq, Phivec, TFvec, uvec_allfreq, wvec_proc, zvec_rec)
    """
    FLAG_SKEMPTON, nw_proc, fRicker, Rhof, Kf, n, ns, shift_z, zn_proc, Kappa0_vec, Kvec, CT0vec, Evec, Vpvec, Vsvec, uEvec_allfreq, Phivec, TFvec, uvec_allfreq, wvec_proc, zvec_rec= argsList
    fdata = np.zeros((zvec_rec.size , ns), dtype=np.complex64)
    tempsInt = 0
    for irec in range(zvec_rec.size) :
        znow = zvec_rec[irec]+shift_z
        #Detecting a layer number
        tmp = zn_proc-znow
        tmp[tmp<0] = np.inf
        tmp1 = np.min(tmp)
        tmp2 = np.argmin(tmp)
        in_now = tmp2
        if(sum(tmp==np.inf) == tmp.size):
            in_now = (zn_proc.size) #last layer

        U_amp = np.squeeze(uvec_allfreq[0,in_now,:]) #upgoing,layer in_now
        D_amp = np.squeeze(uvec_allfreq[1,in_now,:]) #downgoing,layer in_now


        if(Kappa0_vec[in_now] != 0):
            #Porous layer effects: Calculating the function PHI of Ionov
            #   and correponding tube-wave velocities
            #Note: all variables related to the porous layer assumes MATLAB-FT
            tmp = np.sqrt(1j*wvec_proc*TFvec[in_now]) #be carefull INF when kappa0 = 0
            tmp = tmp.astype(np.complex64)
            PHI_IONOV = tmp**(-1)*besselk(1,tmp)/besselk(0,tmp) #be careful NaN when kappa0 = 0

            #Tube-wave velocity including the porous-layer effects
            CTvec = CT0vec[in_now]*np.sqrt(1/(1+2*Phivec[in_now]*Rhof/Kf*CT0vec[in_now]**2*PHI_IONOV)) #this is a frequency-vector version.

            CTvec = np.conj(CTvec) #MATLAB FT -> Aki-Richards FT
            kcvec = wvec_proc/CTvec
        else :
            kcvec = wvec_proc/CT0vec[in_now]

        phi = U_amp.T*np.exp(-1j*kcvec*znow)+D_amp.T*np.exp(1j*kcvec*znow) #displacement potential (note: kc homogenous)
        tmpdata = Rhof*wvec_proc**2*phi #pressure
        #Evalauting the discontinuities (delta_p and delta_vz)
        #due to external effective stress.
        #Note:multiple-freq vector version    
        #I am at z1 = zvec(irec). z0 is the nearest boundary in the negative z direction
        if(in_now == 0):
            z1 = znow 
            z0 = zn_proc[0]
        elif(in_now == n-1):
            z1 = znow 
            z0 = zn_proc[n-2]
        else :
            z1 = znow 
            z0 = zn_proc[in_now-1]
        #I am at in = in_now
        k1vec = kcvec    
        kp1vec = wvec_proc/Vpvec[in_now]
        E1 = Evec[in_now]
        Vp_now = Vpvec[in_now]
        Vs_now = Vsvec[in_now]
        vieuxTemps = time.time()
        with np.errstate(divide='ignore', invalid='ignore'):
        #Integration element I1:
            deltaZ = z0 - z1
            I1 = np.exp(1j*kp1vec*z1)/1j*(-np.expm1(1j*(-k1vec + kp1vec)*deltaZ)/(-k1vec + kp1vec) + np.expm1(1j*(k1vec + kp1vec)*deltaZ)/(k1vec + kp1vec))
            #I1 = np.exp(1j*k1vec*z1)/1j/(-k1vec+kp1vec)*(np.exp(1j*(-k1vec+kp1vec)*z1)-np.exp(1j*(-k1vec+kp1vec)*z0))-np.exp(-1j*k1vec*z1)/1j/(k1vec+kp1vec)*(np.exp(1j*(k1vec+kp1vec)*z1)-np.exp(1j*(k1vec+kp1vec)*z0))
        #Integration element I2:
            I2 = np.exp(-1j*kp1vec*z1)/1j*(-np.expm1(-1j*(k1vec + kp1vec)*deltaZ)/(-k1vec - kp1vec) + np.expm1(1j*(k1vec - kp1vec)*deltaZ)/(k1vec - kp1vec))
            # I2 = np.exp(1j*k1vec*z1)/1j/(-k1vec-kp1vec)*(np.exp(1j*(-k1vec-kp1vec)*z1)-np.exp(1j*(-k1vec-kp1vec)*z0))-np.exp(-1j*k1vec*z1)/1j/(k1vec-kp1vec)*(np.exp(1j*(k1vec-kp1vec)*z1)-np.exp(1j*(k1vec-kp1vec)*z0))
        #Integration element I3:
            # I3 = np.exp(1j*k1vec*z1)/1j/(-k1vec+kp1vec)*(np.exp(1j*(-k1vec+kp1vec)*z1)-np.exp(1j*(-k1vec+kp1vec)*z0))+np.exp(-1j*k1vec*z1)/1j/(k1vec+kp1vec)*(np.exp(1j*(k1vec+kp1vec)*z1)-np.exp(1j*(k1vec+kp1vec)*z0))
        #Integration element I4:
            # I4 = np.exp(1j*k1vec*z1)/1j/(-k1vec-kp1vec)*(np.exp(1j*(-k1vec-kp1vec)*z1)-np.exp(1j*(-k1vec-kp1vec)*z0))+np.exp(-1j*k1vec*z1)/1j/(k1vec-kp1vec)*(np.exp(1j*(k1vec-kp1vec)*z1)-np.exp(1j*(k1vec-kp1vec)*z0))
        tempsInt += time.time() - vieuxTemps
        if(Kappa0_vec[in_now] != 0):
            CT_now = CTvec #vector
        else:
            CT_now = CT0vec[in_now] #scalar

        #---the discontinuities (delta_p and delta_vz) due to elastic wave---
        U_Eamp = np.squeeze(uEvec_allfreq[0,in_now,:]) #upgoing elastic wave,layer in_now
        D_Eamp = np.squeeze(uEvec_allfreq[1,in_now,:]) #downgoing elastic wave,layer in_now

        U_Eamp = U_Eamp.T
        D_Eamp = D_Eamp.T

        with np.errstate(divide='ignore', invalid='ignore'):
            A1P = wvec_proc**2/kp1vec*(1/(2*Vs_now**2)-1/Vp_now**2)
        delta_p_A = -1j*wvec_proc*Rhof*CT_now*kp1vec*A1P*(D_Eamp*I1+U_Eamp*I2)
        #the discontinuities (delta_p and delta_vz) due to a porous layer 
        #(PHI_IONOV_now from Aki-Richards FT)
        if(Kappa0_vec[in_now] != 0):
            Porosity_now = Phivec[in_now]
            K_now = Kvec[in_now]

            delta_p_B = Rhof*CT_now*(-1j*wvec_proc*Porosity_now/Kf*np.conj(PHI_IONOV))*wvec_proc**2/Vp_now**2*K_now*(D_Eamp*I1+U_Eamp*I2) #BUGFIX
            ##Option B
            if (FLAG_SKEMPTON):
                    # The following two lines calculates the Skempton coefficient (B)
                    # assuming the given Vp and Vs to be those in the drained condition.
                    # This is not recommed.
    # $$$                 K_d = K_now #assuming Vp and Vs are drained
    # $$$                 B = 1-1/(1+Kf/(Porosity_now*K_d)) #skepmton coefficient
                    # The following line uses the Skempton coefficient (B) 
                    # provided in PE.B (recommed)
                    #TODO : comme le todo d'avant là
                B = getSkemptonCoefficient(Kf, 0.3)
                    #Application of the Skempton coefficient (B)
                delta_p_B = B*delta_p_B
        else:
            delta_p_B = 0
        tmpdata = tmpdata+delta_p_A+delta_p_B
        tmpdata = tmpdata*fRicker[0:nw_proc] #Note: "Unit incident wave" is
                                            # defined in elastic-wave potentials
        fdata[irec,0:nw_proc] = tmpdata

    fdata[:,0] = 0
    fdata = np.conj(fdata); #Aki-Richards FT -> MATLAB FT
    fdata[:, 1:] = fdata[:, 1:] + np.conj(np.flip(fdata[:, 1:], axis=1)) #tour de passe-passe pour avoir la symétrie des données 
    data = (np.fft.ifft(fdata, axis=1)).real
    data_B = data.T
    #print("int")
    #print(tempsInt)
    return data_B


def displayJAAJ():
    print("jaaj")
