import numpy as np
import utils


class Layers:
    def __init__(self, nlayer: int):
        self.nlayer = nlayer
        self.vp = 5000.0*np.ones(nlayer)
        self.vs = 3000.0*np.ones(nlayer)
        self.bulkDensity = 2500.0*np.ones(nlayer)
        self.boundariesDepth = np.zeros(nlayer-1)
        self.boreHoleRadius = 0.055*np.ones(nlayer)
        self.porosity = 0.3*np.ones(nlayer)
        self.staticPermeability = np.zeros(nlayer)


class Fluid:
    def __init__(self):
        self.dynamicViscosity = 1e-3
        self.acousticVelocity = 1500
        self.initialDensity = 1000
        self.bulkModulus = self.initialDensity*self.acousticVelocity**2

    def setFluidProperties(self, dynamicViscosity: float = 1e-3,
                           acousticVelocity: float = 1500,
                           initialDensity: float = 1000):
        self.dynamicViscosity = dynamicViscosity
        self.acousticVelocity = acousticVelocity
        self.initialDensity = initialDensity
        self.bulkModulus = self.initialDensity*self.acousticVelocity**2


class MeasurementConfiguration:
    def __init__(self):
        self.receiverMaxDepth = 50
        self.receiverGap = 0.1
        self.receiverDepth = np.arange(0, self.receiverMaxDepth + self.receiverGap,
                                       self.receiverGap)
        self.dt = 0.25e-4
        self.timeSampleNumber = 16001
        self.sampledTimes = np.arange(0, (self.timeSampleNumber-1)*self.dt + self.dt,
                                      self.dt)
        self.dw = 1/self.sampledTimes[-1]*2*np.pi
        self.frequencySampleNumber = 300
        self.sampledFrequencies = np.arange(0,
                                            ((self.timeSampleNumber-1)*self.dw
                                             + self.dw),
                                            self.dw)[0:self.frequencySampleNumber]
        self.centerFrequency = 200
        self.delay = 1/self.centerFrequency*2
        self.wavelet = np.zeros_like(self.sampledTimes)

    def initializeWavelet(self, waveletFunction=utils.createRickerWavelet):
        self.wavelet = waveletFunction(self.centerFrequency, self.sampledTimes)


class BoreholeConfiguration:
    """
    Pour définir les paramètres autour du puit
    """

    def __init__(self, nlayer: int):
        self.nlayer = nlayer
        self.layers = Layers(nlayer)
        self.fluid = Fluid()
        self.boreHoleSolution = np.array([])  # résultat des calculs

    def solve(self, config: MeasurementConfiguration):
        solver = Solver()
        self.boreHoleSolution = solver.solve(self, config)


class Solver():
    def solve(self, tube: BoreholeConfiguration, meas: MeasurementConfiguration):
        # -- Tube parameters
        rhoTube = tube.layers.bulkDensity  # Rhovec
        kappaTube = tube.layers.staticPermeability  # Kappa0_vec
        phi = tube.layers.porosity  # Phivec
        vs = tube.layers.vs  # Vsvec
        vp = tube.layers.vp  # Vpvec
        zShift = 0  # shift_z
        nlayer = tube.nlayer  # n
        zBoundaries = tube.layers.boundariesDepth  # zn_org = zn_proc
        radius = tube.layers.boreHoleRadius  # rn
        # -- Measurement parameters
        frequencies = meas.sampledFrequencies  # wvec_proc
        nFrequencies = meas.frequencySampleNumber  # nw_proc
        nTimes = meas.timeSampleNumber  # ns
        wavelet = meas.wavelet  # fRicker
        zMeas = meas.receiverDepth
        # -- Fluid parameters
        nuDyn = tube.fluid.dynamicViscosity  # nu_dyn
        vf = tube.fluid.acousticVelocity  # Vf
        kf = tube.fluid.bulkModulus  # Kf
        rhoFluid = tube.fluid.initialDensity  # Rhof

        # uEvec_allfreq
        elasticWavefieldAmplitude = utils.getElasticWavefield(nlayer,
                                                              zBoundaries,
                                                              frequencies,
                                                              rhoTube,
                                                              vp,
                                                              zShift)
        # Necessary elastic moduli and velocities
        E = rhoTube * vs**2 * (3 * vp**2 - 4 * vs**2) / (vp**2 - vs**2)
        mu = rhoTube * vs**2
        lameCst = rhoTube * vp**2 - 2 * mu  # lambda_vec
        k = lameCst + 2/3 * mu
        ct0 = np.sqrt(vf**2 / (1 + rhoFluid * vf**2 / (rhoTube * vs**2)))
        # Porous-layer effects: Diffusivity (Ionov, eq. A3)
        diffusivity = np.sqrt(kappaTube * kf / (nuDyn * phi))
        relativeDiffusivity = radius**2 / diffusivity**2  # TFvec

        argsList = (kf, nlayer, rhoFluid, radius, elasticWavefieldAmplitude,
                    nFrequencies, frequencies, zBoundaries, ct0, E, kappaTube,
                    k, phi, relativeDiffusivity, vp, vs)
        for i,arg in enumerate(argsList):
            print("somon ?  " + str(i))
            print(arg)
            input()

        tubewaveAmplitudes = utils.getFluidResponse(argsList)  # uvec_allfreq

        argsList = (0, nFrequencies, wavelet, rhoFluid, kf, nlayer,
                    nTimes, 0, zBoundaries, kappaTube, k, ct0,
                    E, vp, vs, elasticWavefieldAmplitude, phi,
                    relativeDiffusivity, tubewaveAmplitudes, frequencies,
                    zMeas)
        solution = utils.getTimeDomainWaveformBorehole(argsList)
        return solution
