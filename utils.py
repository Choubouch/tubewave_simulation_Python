import numpy as np

def createWavelet(f0, tvec):
    """
    Parameters :
        f0
        tvec

    Returns :
        fRicker
    """
    delay = 1/f0**2
    Ricker = 2 * np.pi**2 * f0**2 * (1.0 - 2.0 * (np.pi**2) * (f0**2) * ((tvec - delay)**2)) * np.exp(-(np.pi**2) * (f0**2) * ((tvec - delay)**2))
    Ricker = Ricker/np.max(np.abs(Ricker))
    fRicker = np.conj(np.fft.fft(Ricker))
    return fRicker

    
def createMMatrixElasticWavefield(n, kp1, kp2, z1, rho1, rho2):
    """
    Parameters :
        n, kp1, kp2, z1, rho1, rho2
    
    Returns :
        Mn and MT matrix
    """
    #TODO : enlever les boucles for
    Mn = np.zeros((2,2,n-1), dtype=np.complex128)
    for i_n in range(0, n-1):
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
    return Mn, MT

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

def createSnMatrix(argsList):
    """
    NB : Cy_square, E1 removed (useless)
    """
    n, w, Rhof, Kf, r1, r2, k1, z0, z1, kp1, Vp_now, Vs_now, CT_now, Porosity_now, PHI_IONOV_now, K_now, Kappa0_now, U_Eamp, D_Eamp = argsList
    Sn = np.zeros((2, n-1), dtype=np.complex128)
    for i_n in range(n-1):      
        if i_n == 0 : #gneu gneu les indices sont shiftés de 1
            I1 = 0
            I2 = 0
            I3 = 0
            I4 = 0
        else :
            I1 = np.exp(1j*k1*z1)/1j/(-k1+kp1)*(np.exp(1j*(-k1+kp1)*z1)-np.exp(1j*(-k1+kp1)*z0))-np.exp(-1j*k1*z1)/1j/(k1+kp1)*(np.exp(1j*(k1+kp1)*z1)-np.exp(1j*(k1+kp1)*z0))
            I2 = np.exp(1j*k1*z1)/1j/(-k1-kp1)*(np.exp(1j*(-k1-kp1)*z1)-np.exp(1j*(-k1-kp1)*z0))-np.exp(-1j*k1*z1)/1j/(k1-kp1)*(np.exp(1j*(k1-kp1)*z1)-np.exp(1j*(k1-kp1)*z0))
            I3 = np.exp(1j*k1*z1)/1j/(-k1+kp1)*(np.exp(1j*(-k1+kp1)*z1)-np.exp(1j*(-k1+kp1)*z0))+np.exp(-1j*k1*z1)/1j/(k1+kp1)*(np.exp(1j*(k1+kp1)*z1)-np.exp(1j*(k1+kp1)*z0))
            I4 = np.exp(1j*k1*z1)/1j/(-k1-kp1)*(np.exp(1j*(-k1-kp1)*z1)-np.exp(1j*(-k1-kp1)*z0))+np.exp(-1j*k1*z1)/1j/(k1-kp1)*(np.exp(1j*(k1-kp1)*z1)-np.exp(1j*(k1-kp1)*z0))

        A1P = w**2/kp1*(1/(2*Vs_now**2)-1/Vp_now**2)
        delta_p_A = -1j*w*Rhof*CT_now*kp1*A1P*(D_Eamp*I1+U_Eamp*I2)
        delta_vz_A = -1j*w*kp1*A1P*(D_Eamp*I3+U_Eamp*I4)

        if (Kappa0_now !=0):
            delta_p_B = Rhof*CT_now*(-1j*w*Porosity_now/Kf*PHI_IONOV_now)*w**2/Vp_now**2*K_now*(D_Eamp*I1+U_Eamp*I2)
            delta_vz_B = (-1j*w*Porosity_now/Kf*PHI_IONOV_now)*w**2/Vp_now**2*K_now*(D_Eamp*I3+U_Eamp*I4)
            
            #TODO : traduire le fichier sur skempton 
            #if(FLAG_SKEMPTON):
            #    B = PE.B
            #    delta_p_B = B*delta_p_B
            #    delta_vz_B = B*delta_vz_B   
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
    return Sn
