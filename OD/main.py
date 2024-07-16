import numpy as np
import odlib
from Observation import Observation
from OrbitDetermination import OrbitDetermination
from EphemerisGeneration import EphemerisGeneration
from GaussMethod import GaussMethod


myGauss = GaussMethod((
    Observation(np.radians(odlib.HMS2deg(15, 48, 6.22)), np.radians(odlib.DMS2deg(-2, 46, 44.3)), 2460118.708333333),
    Observation(np.radians(odlib.HMS2deg(15, 50, 18.27)), np.radians(odlib.DMS2deg(1, 12, 55.0 )), 2460125.708333333),
    Observation(np.radians(odlib.HMS2deg(15, 57, 23.59)), np.radians(odlib.DMS2deg(6, 7, 24.6)), 2460132.708333333)
    ))


myGauss.iterative_honing(1e-10)
print(myGauss.get_equatorial_vectors())
myOD = OrbitDetermination(*myGauss.get_ecplitic_vectors(), 2460125.708333333)
myOD.calc_orbital_elements()
print(myOD.get_orbital_elements_dict())
myEphem = EphemerisGeneration(*myOD.get_orbital_elements())
print(np.degrees(myEphem.get_RA_DEC(2460511.75)[0]), np.degrees(myEphem.get_RA_DEC(2460511.75)[1]))