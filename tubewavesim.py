import numpy as np
import utils
import copy
import os


class Layer:
    def __init__(self, kind: str, zTop: float, zBottom: float, porosity: float,
                 staticPermeability: float, bulkDensity: float,
                 vp: float, vs: float):
        self.kind = kind
        self.zTop = zTop
        self.zBottom = zBottom
        self.porosity = porosity
        self.staticPermeability = staticPermeability
        self.vp = vp
        self.vs = vs
        self.bulkDensity = bulkDensity


class LayerStack:
    def __init__(self):
        self.layerList = []
        self.size = 0
        self.backGroundLayer: Layer
        self.radiusList = []
        self.defaultRadius = -1

    def appendLayer(self, kind: str, zTop: float, zBottom: float, porosity: float,
                    staticPermeability: float, bulkDensity: float,
                    vp: float, vs: float):
        newLayer = Layer(kind, zTop, zBottom, porosity,
                         staticPermeability, bulkDensity, vp, vs)
        self.layerList.append(newLayer)
        self.size += 1

    def clearStack(self):
        self.layerList = []
        self.size = 0

    def sort(self):
        newLayerList = []
        newLayerList = sorted(self.layerList, key=lambda L: L.zTop)
        self.backGroundLayer = newLayerList.pop(0)
        self.size -= 1
        for i in range(self.size-1):
            if (newLayerList[i].zBottom > newLayerList[i+1].zTop):
                print("Layer overlap between : ")
                print("Layer "+str(i) + "  zTop = " + str(newLayerList[i].zTop) + "  zBottom = " + str(newLayerList[i].zTop))
                print("Layer "+str(i+1) + "  zTop = " + str(newLayerList[i+1].zTop) + "  zBottom = " + str(newLayerList[i+1].zTop))
        self.layerList = newLayerList
        self.radiusList = sorted(self.radiusList, key=lambda rad: rad["zTop"])

    def addRadiusChange(self, radius, zTop, zBottom):
        radiusChange = {"radius": radius, "zTop": zTop, "zBottom": zBottom}
        self.radiusList.append(radiusChange)

    def setDefaultRadius(self, radius):
        self.defaultRadius = radius


class LayeredMedia:
    def __init__(self, nlayer: int = 5, vp: float = 5000., vs: float = 3000.,
                 bulkDensity: float = 2500., radius: float = 0.055,
                 porosity: float = 0.3):
        """
            Parameters :
                nlayer should be greater than 2
                default values correspond to those in Minato's article
        """
        self.nlayer = nlayer
        self.vp = vp*np.ones(nlayer)
        self.vs = vs*np.ones(nlayer)
        self.bulkDensity = bulkDensity*np.ones(nlayer)
        self.boundariesDepth = np.zeros(nlayer-1)
        self.boreHoleRadius = radius*np.ones(nlayer)
        self.porosity = porosity*np.ones(nlayer)
        self.staticPermeability = np.zeros(nlayer)
        self.stack = LayerStack()

    def clear(self):
        self.nlayer = 0
        self.vp = np.empty(0)
        self.vs = np.empty(0)
        self.bulkDensity = np.empty(0)
        self.boundariesDepth = np.empty(0)
        self.boreHoleRadius = np.empty(0)
        self.porosity = np.empty(0)
        self.staticPermeability = np.empty(0)

    def __appendLayer(self, layer: Layer):
        # layerSubdivisionCstRadius = []
        # for radius in self.stack.radiusList:
        #     if radius["zTop"]  > layer.zTop and radius["zTop"] < layer.zBottom:
        #         subLayer = copy.deepcopy(layer)
        #         subLayer.zTop = radius["zTop"]
        #         subLayer.zBottom = radius["zBottom"]
        #         layerSubdivisionCstRadius.append(subLayer)

        self.vp = np.append(self.vp, layer.vp)
        self.vs = np.append(self.vs, layer.vs)
        self.bulkDensity = np.append(self.bulkDensity, layer.bulkDensity)
        self.porosity = np.append(self.porosity, layer.porosity)
        self.staticPermeability = np.append(self.staticPermeability, layer.staticPermeability)
        if layer.zTop == -np.inf:
            return 0
        if layer.zBottom == np.inf:
            return 0
        if (self.boundariesDepth.size != 0 and layer.zTop == self.boundariesDepth[-1]):
            self.boundariesDepth = np.append(self.boundariesDepth, layer.zBottom)
            return 0
        self.boundariesDepth = np.append(self.boundariesDepth, [layer.zTop, layer.zBottom])

    def __appendRadius(self, radius):
        self.boreHoleRadius = np.append(self.boreHoleRadius, radius)

    def createSubdivisionCstRadius(self, continuousLayerList):
        # Layers and radius sorted
        newLayerSubdivision = []
        newRadiusSubdivision = []
        unvisitedLayerHeap = []
        unvisitedLayerHeap = copy.deepcopy(continuousLayerList)
        unvisitedLayerHeap.reverse()
        while len(unvisitedLayerHeap) > 0:
            currentLayer = unvisitedLayerHeap.pop(-1)
            # print(currentLayer.kind, currentLayer.zTop, currentLayer.zBottom)
            # input()
            radiusFound = 0
            for radius in self.stack.radiusList:
                # print(radius)
                # input()
                if radius["zTop"] > currentLayer.zTop and radius["zTop"] < currentLayer.zBottom:
                    LayerUp = copy.deepcopy(currentLayer)
                    LayerUp.zBottom = radius["zTop"]
                    LayerDown = copy.deepcopy(currentLayer)
                    LayerDown.zTop = radius["zTop"]
                    newLayerSubdivision.append(LayerUp)
                    newRadiusSubdivision.append(self.stack.defaultRadius)
                    unvisitedLayerHeap.append(LayerDown)
                    radiusFound = 1
                    # print("1")
                    # print(LayerUp.kind, LayerUp.zTop, LayerUp.zBottom)
                    # input()
                    break

                if radius["zBottom"] > currentLayer.zTop and radius["zBottom"] < currentLayer.zBottom:
                    LayerUp = copy.deepcopy(currentLayer)
                    LayerUp.zBottom = radius["zBottom"]
                    LayerDown = copy.deepcopy(currentLayer)
                    LayerDown.zTop = radius["zBottom"]
                    newLayerSubdivision.append(LayerUp)
                    newRadiusSubdivision.append(radius["radius"])
                    unvisitedLayerHeap.append(LayerDown)
                    radiusFound = 1
                    # print("2")
                    # print(LayerUp.kind, LayerUp.zTop, LayerUp.zBottom)
                    # input()
                    break

                if radius["zBottom"] >= currentLayer.zBottom and radius["zTop"] <= currentLayer.zTop:
                    newLayerSubdivision.append(currentLayer)
                    newRadiusSubdivision.append(radius["radius"])
                    radiusFound = 1
                    # print("3")
                    # print(currentLayer.kind, currentLayer.zTop, currentLayer.zBottom)
                    # input()
                    break
                # print(4)
                # input()
            if (radiusFound == 0):
                newLayerSubdivision.append(currentLayer)
                newRadiusSubdivision.append(self.stack.defaultRadius)
        return (newLayerSubdivision, newRadiusSubdivision)

    def createContinuousLayers(self):
        continuousLayerList = []
        backGroundLayer = self.stack.backGroundLayer
        firstInfill = copy.deepcopy(backGroundLayer)
        firstInfill.zBottom = self.stack.layerList[0].zTop
        continuousLayerList.append(firstInfill)
        for i in range(self.stack.size):
            currentLayer = self.stack.layerList[i]
            # Need to fill the gap between two layers
            if (currentLayer.zTop > continuousLayerList[-1].zBottom):
                infillLayer = copy.copy(backGroundLayer)
                infillLayer.zTop = continuousLayerList[-1].zBottom
                infillLayer.zBottom = currentLayer.zTop
                continuousLayerList.append(infillLayer)
            continuousLayerList.append(currentLayer)
        if (continuousLayerList[-1].zBottom == np.inf):  # no need to fill
            return continuousLayerList
        lastInfill = copy.deepcopy(backGroundLayer)
        lastInfill.zTop = continuousLayerList[-1].zBottom
        continuousLayerList.append(lastInfill)
        return continuousLayerList

    def buildLayers(self):
        self.clear()
        self.stack.sort()
        continuousLayerList = self.createContinuousLayers()
        subdividedLayers, subdividedRadius = self.createSubdivisionCstRadius(continuousLayerList)
        # for i in range(len(subdividedRadius)):
            # print(i, subdividedRadius[i], subdividedLayers[i].kind, subdividedLayers[i].zTop, subdividedLayers[i].zBottom)
        # input()
        backGroundLayer = self.stack.backGroundLayer
        # self.__appendLayer(backGroundLayer)
        for i in range(len(subdividedLayers)):
            self.__appendLayer(subdividedLayers[i])
            self.__appendRadius(subdividedRadius[i])
            self.nlayer = self.vp.size

#        for i in range(self.stack.size):
#            currentLayer = self.stack.layerList[i]
#            # Need to fill the gap between two layers
#            if (self.boundariesDepth.size != 0 and currentLayer.zTop > self.boundariesDepth[-1]):
#                infillLayer = copy.copy(backGroundLayer)
#                infillLayer.zTop = self.boundariesDepth[-1]
#                infillLayer.zBottom = currentLayer.zTop
#                self.__appendLayer(infillLayer)
#            self.__appendLayer(currentLayer)
#        self.__appendLayer(backGroundLayer)
#        self.nlayer = self.vp.size


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
        self.receiverMinDepth = 0
        self.receiverGap = 0.1
        self.receiverDepth = np.arange(self.receiverMinDepth, self.receiverMaxDepth + self.receiverGap,
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

    def setDepthBoundaries(self, maxDepth: float, minDepth: float, receiverGap: float = 0.1):
        self.receiverMaxDepth = maxDepth
        self.receiverMinDepth = minDepth
        self.receiverGap = receiverGap
        self.receiverDepth = np.arange(self.receiverMinDepth, self.receiverMaxDepth + self.receiverGap,
                                       self.receiverGap)


class BoreholeConfiguration:
    """
    Pour définir les paramètres autour du puit
    """

    def __init__(self, nlayer: int = 5):
        self.layers = LayeredMedia(nlayer)
        self.fluid = Fluid()
        self.boreHoleSolution = np.array([])  # résultat des calculs

    def solve(self, config: MeasurementConfiguration):
        solver = Solver()
        self.boreHoleSolution = solver.solve(self, config)

    def getFluidPressureData(self, config: MeasurementConfiguration, depth):
        """
        Param : 
            depth : float or array of floats
        """
        if not (isinstance(depth, np.ndarray)):
            depth = np.array(depth)
        # Les quelques lignes suivantes font ce qu'on veut qu'elles fassent
        # ie récupèrent les indices correspondants aux profondeurs de l'array depth
        depth = depth[:, np.newaxis]
        condition = np.isclose(config.receiverDepth, depth)
        indices = np.where(condition)[1]
        return self.boreHoleSolution.T[[indices]][0]

    def showConfiguration(self, interrupt=False, clearTerminal=False):
        if clearTerminal:
            os.system("clear")
        print(-np.inf, end="")
        for i in range(self.layers.nlayer):
            print("  ----------------------------------------------------------")
            print("|", end="")
            print(" Vp = ", end="")
            print(self.layers.vp[i], end="")
            print("    ", end="")
            print(" Vs = ", end="")
            print(self.layers.vs[i])

            print("|", end="")
            print(" Bulk Density =", end="")
            print(self.layers.bulkDensity[i], end="")
            print("    ", end="")
            print(" Bore Hole Radius =", end="")
            print(self.layers.boreHoleRadius[i])

            print("|", end="")
            print(" Porosity =", end="")
            print(self.layers.porosity[i], end="")
            print("    ", end="")
            print(" Static Permeability =", end="")
            print(self.layers.staticPermeability[i])

            if (i != self.layers.nlayer-1):
                print(self.layers.boundariesDepth[i], end="")

            else:
                print(np.inf, end="")
                print("  ----------------------------------------------------------")
        if interrupt:
            input("\n \n Press Enter to continue")




            
#  self.nlayer = nlayer
#         self.vp = vp*np.ones(nlayer)
#         self.vs = vs*np.ones(nlayer)
#         self.bulkDensity = bulkDensity*np.ones(nlayer)
#         self.boundariesDepth = np.zeros(nlayer-1)
#         self.boreHoleRadius = radius*np.ones(nlayer)
#         self.porosity = porosity*np.ones(nlayer)
#         self.staticPermeability = np.zeros(nlayer)
# 




class Solver():
    def solve(self, tube: BoreholeConfiguration, meas: MeasurementConfiguration):
        # -- Tube parameters
        rhoTube = tube.layers.bulkDensity  # Rhovec
        kappaTube = tube.layers.staticPermeability  # Kappa0_vec
        phi = tube.layers.porosity  # Phivec
        vs = tube.layers.vs  # Vsvec
        vp = tube.layers.vp  # Vpvec
        zShift = 0  # shift_z
        nlayer = tube.layers.nlayer  # n
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
        with np.errstate(divide='ignore'):
            relativeDiffusivity = radius**2 / diffusivity**2  # TFvec

        argsList = (kf, nlayer, rhoFluid, radius, elasticWavefieldAmplitude,
                    nFrequencies, frequencies, zBoundaries, ct0, E, kappaTube,
                    k, phi, relativeDiffusivity, vp, vs)
        tubewaveAmplitudes = utils.getFluidResponse(argsList)  # uvec_allfreq

        argsList = (1, nFrequencies, wavelet, rhoFluid, kf, nlayer,
                    nTimes, 0, zBoundaries, kappaTube, k, ct0,
                    E, vp, vs, elasticWavefieldAmplitude, phi,
                    relativeDiffusivity, tubewaveAmplitudes, frequencies,
                    zMeas)
        solution = utils.getTimeDomainWaveformBorehole(argsList)
        return solution


# stack = LayerStack()
# stack.appendLayer("toto", 18, 115, 2, 1, 1, 1, 1)
# stack.appendLayer("jaaj", 16, 18, 2, 1, 1, 1, 1)
# stack.appendLayer("jouj", 1, 17, 2, 1, 1, 1, 1)
#
# stack.appendLayer("juuj", -np.inf, np.inf, 2, 1, 1, 1, 1)
# stack.sort()

# edia = LayeredMedia()
# media.stack.appendLayer("bg", -np.inf, np.inf, 0.3, 0, 2500, 5000, 3000)
# media.stack.appendLayer("elastic", 0, 19.5, 0.5, 0, 2500, 5000, 3000)
# media.stack.setDefaultRadius(0.02)
# media.stack.addRadiusChange(0.05, -1, 5)
# media.stack.addRadiusChange(0.01, 6, 25)
# media.stack.addRadiusChange(0.1, 30, 50)
# media.stack.appendLayer("elastic", 20.5, 50, 0.3, 0, 2500, 5000, 3000)
# media.stack.appendLayer("porous", 19.5, 20.5, 0.3, 1*9.869e-13, 2500, 5000, 3000)
# media.buildLayers()
# print(media.boundariesDepth)
# print(media.porosity)
# print(media.staticPermeability)
#
# print("porosity")
# print(media.porosity)
