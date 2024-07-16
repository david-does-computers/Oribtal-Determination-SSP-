import numpy as np
from astroquery.jplhorizons import Horizons
import odlib
from Observation import Observation

class GaussMethod:
    GAUSSIAN_DAY = 182.62844915 / np.pi      # Solar days per Gaussian Day
    K = 0.0172020989484                      # Gaussian gravitational constant

    def __init__(self, observations: tuple[Observation]) -> None:
        """
        Initializes the Method of Gauss object

        :param observations: tuple of length 3 housing the observations to be used 
        """
        self.sun = Horizons(id='@sun', location='500')

        self.observations = observations

        self.taus = np.array([                                                      #Look to Dubson OD Packet for formatting
            self.K*(observations[2].get_time() - observations[0].get_time()), 
            self.K*(observations[0].get_time() - observations[1].get_time()), 
            self.K*(observations[2].get_time() - observations[1].get_time())])
        
        self.a1 = self.taus[2] / self.taus[0]
        self.a3 = -self.taus[1] / self.taus[0]

        self._calc_D_values()
        self._calc_rho_vectors()

        self.r_vectors = np.array([
            self.rhos[0]*self.observations[0].get_rho_hat() - self.observations[0].get_sun_vector(),
            self.rhos[1]*self.observations[1].get_rho_hat() - self.observations[1].get_sun_vector(),
            self.rhos[2]*self.observations[2].get_rho_hat() - self.observations[2].get_sun_vector()
        ])

        self.rdot: np.ndarray = ((self.observations[2].get_time() - self.observations[1].get_time())/(self.observations[1].get_time() - self.observations[0].get_time())*(self.r_vectors[1]-self.r_vectors[0]) #Velcoity vector of middle observation
        + (self.observations[1].get_time() - self.observations[0].get_time())/(self.observations[2].get_time() - self.observations[1].get_time())*(self.r_vectors[2]-self.r_vectors[1])) / (self.observations[2].get_time() - self.observations[0].get_time())

        self.rdot /= self.K # Accomodate for Gaussian Days

        self._calc_f_g_values()

        self._calc_a_values()


    def iterative_honing(self, tolerance: np.float64=1e-10):
        """
        Iteratively recalculate until chnage in position vector is less than tolerance

        :param tolerance: threshold which dictates the maximum change in positoin vector before quitting iteration
        """
        self.prev_r = 1.1 * self.r_vectors[1]
        while np.abs(1 -np.linalg.norm(self.prev_r)/odlib.magnitude(self.r_vectors[1])) > tolerance: #TODO: Increased efficiency using magnitude squared?
            self._calc_rho_vectors()
            self._calc_r_vectors()
            self._calc_f_g_values()
            self._calc_a_values()

    def get_equatorial_vectors(self):
        return self.r_vectors[1], self.rdot
    
    def get_ecplitic_vectors(self, eq_obliquity: np.float64=0.409092804):
        return np.array([
            [1, 0, 0],
            [0, np.cos(eq_obliquity), np.sin(eq_obliquity)],
            [0, -np.sin(eq_obliquity), np.cos(eq_obliquity)]
        ]) @ self.r_vectors[1], np.array([
            [1, 0, 0],
            [0, np.cos(eq_obliquity), np.sin(eq_obliquity)],
            [0, -np.sin(eq_obliquity), np.cos(eq_obliquity)]
        ]) @ self.rdot

    def _calc_D_values(self):
        """
        Calculates the D values for the set of observations
        """
        self.D_0 = odlib.triple(self.observations[0].get_rho_hat(), self.observations[1].get_rho_hat(), self.observations[2].get_rho_hat())

        self.D_values = np.zeros((3, 3))

        for i in range(3):
            self.D_values[0, i] = np.array([
                odlib.triple(self.observations[2].get_rho_hat(), self.observations[i].get_sun_vector(), self.observations[1].get_rho_hat())
            ])
            self.D_values[1, i] = np.array([
                odlib.triple(self.observations[2].get_rho_hat(), self.observations[0].get_rho_hat(), self.observations[i].get_sun_vector())
            ])
            self.D_values[2, i] = np.array([
                odlib.triple(self.observations[0].get_rho_hat(), self.observations[1].get_rho_hat(), self.observations[i].get_sun_vector())
            ])

    def _calc_rho_vectors(self):
        """
        Calclulates the rho (earth-asteroid) vectors and updates the object variables
        """
        self.rhos = np.array([
            (self.a1 * self.D_values[0,0] - self.D_values[0,1] + self.a3 * self.D_values[0,2]) / self.a1 / self.D_0, 
            - (self.a1 * self.D_values[1,0] - self.D_values[1,1] + self.a3 * self.D_values[1,2]) / self.D_0,
            (self.a1 * self.D_values[2,0] - self.D_values[2,1] + self.a3 * self.D_values[2,2]) / self.a3 / self.D_0
        ])

    def _calc_f_g_values(self):
        """
        Calculates and updates the f and g values
        """
        self.f_g_values = np.array([
            [self._f(self.taus[1]), self._f(self.taus[2])],
            [self._g(self.taus[1]), self._g(self.taus[2])]
        ])

    def _calc_r_vectors(self):
        """
        Calculates the r (sun-asteroid) vectors and the velocity vector fo the second observation updates the object variables
        """
        self.prev_r = odlib.magnitude(self.r_vectors[1])
        for i in range(3):
            self.r_vectors[i] = self.rhos[i]*self.observations[i].get_rho_hat() - self.observations[i].get_sun_vector()

        self.r_vectors[1] = (self.f_g_values[1,1]*self.r_vectors[0]-self.f_g_values[1,0]*self.r_vectors[2]) / (self.f_g_values[0,0]*self.f_g_values[1,1] - self.f_g_values[0,1]*self.f_g_values[1,0])

        self.rdot = (self.f_g_values[0,1]* self.r_vectors[0] - self.f_g_values[0,0]*self.r_vectors[2]) / (self.f_g_values[0,1]*self.f_g_values[1,0] - self.f_g_values[0,0]*self.f_g_values[1,1])


    def _f(self, tau: np.float64):
        """
        Returns the fourth degree f series of tau

        :param tau: gaussian time [GD]
        """
        r = odlib.magnitude(self.r_vectors[1]) # Magnitude of r-vector
        rdot = odlib.magnitude(self.rdot)# Magnitude of rdot-vector

        return 1 - (0.5 / r**3) * tau**2 + (self.rdot.dot(self.r_vectors[1]) / 2 / r**5) * tau**3 + ((3*((rdot / r)**2 - 1/r**3)-15*(self.r_vectors[1].dot(self.rdot)/r**2)**2 + 1/r**3) / 24 / r ** 3) * tau**4

    def _g(self, tau: np.float64):
        """
        Returns the fourth degree g series of tau

        :param tau: gaussian time [GD]
        """
        r = odlib.magnitude(self.r_vectors[1]) # Magnitude of r-vector

        return tau - (1 / 6 / r**3) * tau **3 + (self.rdot.dot(self.r_vectors[1]) / 4 / r**5) * tau ** 4
    
    def _calc_a_values(self):
        """
        Calculates and updates the object's a values
        """
        denominator = (self.f_g_values[0,0]*self.f_g_values[1,1]-self.f_g_values[0,1]*self.f_g_values[1,0])
        self.a1 = self.f_g_values[1,1] / denominator
        self.a3 = -self.f_g_values[1,0] / denominator


    def _get_sun_vector(self, time: np.float64) -> np.ndarray:
        """
        Gets the sun vector from the center of the earth

        :param time: time [JD]
        """
        self.sun.epochs = time
        vecs = self.sun.vectors(refplane="earth")
        return np.array([float(vecs["x"]), float(vecs["y"]), float(vecs["z"])], dtype=np.float64)