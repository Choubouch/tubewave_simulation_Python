import numpy as np
from scipy.special import kv as besselk
from typing import List

def createRickerWavelet(f0, tvec):
    """
    Parameters :
        f0
        tvec

    Returns :
        fRicker
    """
    print("f0" + str(f0))
    print("Tvec 0-20")
    input()
    delay = 1/f0*2
    Ricker = 2 * np.pi**2 * f0**2 * (1.0 - 2.0 * (np.pi**2) * (f0**2) * ((tvec - delay)**2)) * np.exp(-(np.pi**2) * (f0**2) * ((tvec - delay)**2))
    print("Ricker")
    print(Ricker[:20])
    input()
    Ricker = Ricker/np.max(np.abs(Ricker))
    fRicker = np.conj(np.fft.fft(Ricker))
    return fRicker

    
def createElasticWavefield(n, zn_proc, wvec_proc, Rhovec, Vpvec, shift_z):
    """
    Parameters :
        n, kp1, kp2, z1, rho1, rho2
    
    Returns :
        Mn and MT matrix
    """
    #TODO : enlever les boucles for
    nw_proc = wvec_proc.size
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

    return uEvec_allfreq

def createMMatrixPressureField(n, k1, k2, z1, r1, r2):
    """
    Parameters :
        n, k1, k2, z1, r1, r2

    Returns :
        Mn

    """
    Mn = np.zeros((2, 2, n-1), dtype=np.complex128)
    for i_n in range(n-1):
        a1=r1**2*k1+r2**2*k2
        a2=r1**2*k1-r2**2*k2
        m11=a1*np.exp(1j*(k1-k2)*z1) 
        m12=a2*np.exp(1j*(k1+k2)*z1) 
        m21=a2*np.exp(-1j*(k1+k2)*z1) 
        m22=a1*np.exp(-1j*(k1-k2)*z1) 

        Mn[:,:,i_n]=1/(2*r1**2*k1)*np.array([[m11, m12], [m21, m22]])
    return Mn

def getTubewavePotential(argsList : List[float]):
    """
    argsList = ( Kf, n, Rhof, rn, uEvec_allfreq, nw_proc, wvec_proc, zn_proc, CT0vec, Evec, Kappa0_vec, Kvec, Phivec, TFvec, Vpvec, Vsvec)
    NB : Cy_square, E1 removed (useless)
    """
    Kf, n, Rhof, rn, uEvec_allfreq, nw_proc, wvec_proc, zn_proc, CT0vec, Evec, Kappa0_vec, Kvec, Phivec, TFvec, Vpvec, Vsvec = argsList
    uvec_allfreq = np.zeros((2, n, nw_proc), dtype=np.complex128)
    for iw in range(1, nw_proc):
        w = wvec_proc[iw]
        kp = w/Vpvec

        tmp = np.sqrt(1j*w*TFvec)
        tmp = tmp.astype(np.complex128)
        PHI_IONOV = (1/tmp) * besselk(1., tmp) / besselk(0., tmp)

        CTvec = CT0vec * np.sqrt(1/(1+2*Phivec*Rhof/Kf*CT0vec**2 * PHI_IONOV))

        CTvec[Kappa0_vec == 0] = CT0vec[Kappa0_vec == 0]
        CTvec = np.conj(CTvec)

        kn = w/CTvec

        Mn = np.zeros((2, 2, n-1), dtype=np.complex128)
        for i_n in range(n-1):
            r1=rn[i_n]
            r2=rn[i_n+1]
            k1=kn[i_n]
            k2=kn[i_n+1]
            z1=zn_proc[i_n]
    
            a1=r1**2*k1+r2**2*k2
            a2=r1**2*k1-r2**2*k2
            
            m11=a1*np.exp(1j*(k1-k2)*z1) 
            m12=a2*np.exp(1j*(k1+k2)*z1) 
            m21=a2*np.exp(-1j*(k1+k2)*z1) 
            m22=a1*np.exp(-1j*(k1-k2)*z1) 

            Mn[:,:,i_n]=1/(2*r1**2*k1)*np.array([[m11, m12], [m21, m22]])

        Sn = np.zeros((2, n-1), dtype=np.complex128)
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
                #if(FLAG_SKEMPTON):
                #    B = PE.B
                #    delta_p_B = B*delta_p_B
                #    delta_vz_B = B*delta_vz_B   
            else :
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
        uvec = np.zeros((2,n), dtype=np.complex128)
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
    fdata = np.zeros((zvec_rec.size , ns), dtype=np.complex128)

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
            tmp = tmp.astype(np.complex128)
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


        #Integration element I1:
        I1 = np.exp(1j*k1vec*z1)/1j/(-k1vec+kp1vec)*(np.exp(1j*(-k1vec+kp1vec)*z1)-np.exp(1j*(-k1vec+kp1vec)*z0))-np.exp(-1j*k1vec*z1)/1j/(k1vec+kp1vec)*(np.exp(1j*(k1vec+kp1vec)*z1)-np.exp(1j*(k1vec+kp1vec)*z0))
        #Integration element I2:
        I2 = np.exp(1j*k1vec*z1)/1j/(-k1vec-kp1vec)*(np.exp(1j*(-k1vec-kp1vec)*z1)-np.exp(1j*(-k1vec-kp1vec)*z0))-np.exp(-1j*k1vec*z1)/1j/(k1vec-kp1vec)*(np.exp(1j*(k1vec-kp1vec)*z1)-np.exp(1j*(k1vec-kp1vec)*z0))
        #Integration element I3:
        I3 = np.exp(1j*k1vec*z1)/1j/(-k1vec+kp1vec)*(np.exp(1j*(-k1vec+kp1vec)*z1)-np.exp(1j*(-k1vec+kp1vec)*z0))+np.exp(-1j*k1vec*z1)/1j/(k1vec+kp1vec)*(np.exp(1j*(k1vec+kp1vec)*z1)-np.exp(1j*(k1vec+kp1vec)*z0))
        #Integration element I4:
        I4 = np.exp(1j*k1vec*z1)/1j/(-k1vec-kp1vec)*(np.exp(1j*(-k1vec-kp1vec)*z1)-np.exp(1j*(-k1vec-kp1vec)*z0))+np.exp(-1j*k1vec*z1)/1j/(k1vec-kp1vec)*(np.exp(1j*(k1vec-kp1vec)*z1)-np.exp(1j*(k1vec-kp1vec)*z0))
        
        if(Kappa0_vec[in_now] != 0):
            CT_now = CTvec #vector
        else:
            CT_now = CT0vec[in_now] #scalar
                    

        #---the discontinuities (delta_p and delta_vz) due to elastic wave---
        U_Eamp = np.squeeze(uEvec_allfreq[0,in_now,:]) #upgoing elastic wave,layer in_now
        D_Eamp = np.squeeze(uEvec_allfreq[1,in_now,:]) #downgoing elastic wave,layer in_now

        U_Eamp = U_Eamp.T
        D_Eamp = D_Eamp.T   

        A1P = wvec_proc**2/kp1vec*(1/(2*Vs_now**2)-1/Vp_now**2)
        delta_p_A = -1j*wvec_proc*Rhof*CT_now*kp1vec*A1P*(D_Eamp*I1+U_Eamp*I2)
        #the discontinuities (delta_p and delta_vz) due to a porous layer 
        #(PHI_IONOV_now from Aki-Richards FT)
        if(Kappa0_vec[in_now] != 0):
            Porosity_now = Phivec[in_now]                
            K_now = Kvec[in_now]

            delta_p_B = Rhof*CT_now*(-1j*wvec_proc*Porosity_now/Kf*np.conj(PHI_IONOV))*wvec_proc**2/Vp_now**2*K_now*(D_Eamp*I1+U_Eamp*I2) #BUGFIX
        
            ##Option B
            if(FLAG_SKEMPTON):
                continue
                    # The following two lines calculates the Skempton coefficient (B)
                    # assuming the given Vp and Vs to be those in the drained condition.
                    # This is not recommed.
    # $$$                 K_d = K_now #assuming Vp and Vs are drained
    # $$$                 B = 1-1/(1+Kf/(Porosity_now*K_d)) #skepmton coefficient
                    # The following line uses the Skempton coefficient (B) 
                    # provided in PE.B (recommed)
                    #TODO : comme le todo d'avant là
                    #B = PE.B
                    #Application of the Skempton coefficient (B)
                    #delta_p_B = B*delta_p_B
        
        else:  
            delta_p_B = 0
        
        
        
        
        tmpdata = tmpdata+delta_p_A+delta_p_B
        tmpdata = tmpdata*fRicker[0:nw_proc] #Note: "Unit incident wave" is
                                            # defined in elastic-wave potentials
        fdata[irec,0:nw_proc] = tmpdata
        
        #input()
        

    fdata[:,0] = 0
    fdata = np.conj(fdata); #Aki-Richards FT -> MATLAB FT
    fdata[:, 1:] = fdata[:, 1:] + np.conj(np.flip(fdata[:, 1:], axis=1)) #tour de passe-passe pour avoir la symétrie des données 
    data = (np.fft.ifft(fdata, axis=1)).real
    data_B = data.T
    return data_B

def displayJAAJ():
    print("jaaj")
