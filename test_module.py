import tubewavesim as ts
import numpy as np
import matplotlib.pyplot as plt


# CALIPER CHANGE
measure1 = ts.MeasurementConfiguration()
measure1.initializeWavelet()
measure1.setDepthBoundaries(20, -20)

caliperChangeTube= ts.BoreholeConfiguration(5)
caliperChangeTube.layers.stack.appendLayer("elastic", -np.inf, np.inf, 0.3, 0, 2500, 4000, 2000)
caliperChangeTube.layers.stack.setDefaultRadius(0.055)
caliperChangeTube.layers.stack.addRadiusChange(0.065, 0, 1)
caliperChangeTube.layers.stack.addRadiusChange(0.065, 1, np.inf)
caliperChangeTube.layers.stack.appendLayer("elastic", -5, 0, 0.3, 0, 2500, 4000, 2000)
caliperChangeTube.layers.buildLayers()
caliperChangeTube.solve(measure1)
plt.imshow(caliperChangeTube.boreHoleSolution.T[::2, ::2])
plt.imshow(caliperChangeTube.boreHoleSolution.T[::2, ::2], vmin=-0.12, vmax=0.12, aspect=1/1000, extent=((measure1.sampledTimes)[0], (measure1.sampledTimes)[-1], measure1.receiverMaxDepth, measure1.receiverMinDepth))
plt.xlim((0, 0.03))
plt.show()

pressure = caliperChangeTube.getFluidPressureData(measure1, np.array([-10]))[0]
plt.plot(measure1.sampledTimes-0.004, pressure)
plt.xlim((0, 0.02))
plt.ylim((-0.11, 0.11))
plt.grid()
plt.show()
# POROUS LAYER 
porousTube = ts.BoreholeConfiguration()
porousTube.layers.stack.appendLayer("elastic", -np.inf, np.inf, 0.3, 0, 2500, 5000, 3000)
porousTube.layers.stack.appendLayer("porous", 0, 0.5, 0.3, 1*9.869e-13, 2500, 5000, 3000)
porousTube.layers.stack.setDefaultRadius(0.055)

porousTube.layers.buildLayers()
porousTube.solve(measure1)
plt.imshow(porousTube.boreHoleSolution.T[::2, ::2])
plt.imshow(porousTube.boreHoleSolution.T[::2, ::2], vmin=-0.03, vmax=0.03, aspect=1/1000, extent=((measure1.sampledTimes)[0], (measure1.sampledTimes)[-1], measure1.receiverMaxDepth, measure1.receiverMinDepth))
plt.xlim((0, 0.03))
plt.show()

# ELASTIC 3 LAYERS 
elasticLayersTube = ts.BoreholeConfiguration()
elasticLayersTube.layers.stack.appendLayer("elastic", -np.inf, np.inf, 0.3, 0, 2500, 5000, 3000)
elasticLayersTube.layers.stack.appendLayer("elastic", 0, 0.5, 0.3, 0, 2500, 3500, 2100)
elasticLayersTube.layers.stack.setDefaultRadius(0.055)
elasticLayersTube.layers.buildLayers()
elasticLayersTube.solve(measure1)
plt.imshow(elasticLayersTube.boreHoleSolution.T[::2, ::2])
plt.imshow(elasticLayersTube.boreHoleSolution.T[::2, ::2], vmin=-0.03, vmax=0.03, aspect=1/1000, extent=((measure1.sampledTimes)[0], (measure1.sampledTimes)[-1], measure1.receiverMaxDepth, measure1.receiverMinDepth))
plt.xlim((0, 0.03))
plt.show()

# A LOT OF LAYERS
sandwich = ts.BoreholeConfiguration()
sandwich.layers.stack.appendLayer("elastic", -np.inf, np.inf, 0.3, 0, 2500, 5000, 3000)
for i in range(20):
    sandwich.layers.stack.appendLayer("elastic", i*20, i*20+1, 0.3, 0, 2500, 3500, 2100)
    sandwich.layers.stack.appendLayer("porous", i*20+10, i*20+11, 0.3, 1*9.869e-13, 2500, 5000, 3000)
sandwich.layers.stack.setDefaultRadius(0.055)
sandwich.layers.buildLayers()

measure2 = ts.MeasurementConfiguration()
measure2.initializeWavelet()
measure2.setDepthBoundaries(500, 0)

sandwich.solve(measure2)
print(sandwich.boreHoleSolution.T[::2, ::2].shape)

plt.imshow(sandwich.boreHoleSolution.T[::2, ::2], vmin=-0.03, vmax=0.03, aspect=1/100, extent=((measure1.sampledTimes)[0], (measure1.sampledTimes)[-1], measure1.receiverMaxDepth, measure1.receiverMinDepth))

plt.show()





