import numpy as np
import odlib
from astroquery.jplhorizons import Horizons

class EphemerisGeneration:
    
    GAUSSIAN_DAY = 182.62844915 / np.pi      # Solar days per Gaussian Day
    K = 0.0172020989484                      # Gaussian gravitational constant

    def __init__(self, semi_major_axis: np.float64, eccentricty: np.float64, inclination: np.float64, long_asc_node: np.float64, arg_peri: np.float64, time_peri: np.float64) -> None:
        """
        Initializes the Ephemeris Generation object using the orbital elements

        :param semi_major_axis: semi-major axis of orbit [AU]
        :param eccentricty: orbital eccentricity 
        :param inclination: orbital inclincation wrt ecliplitic plane [rad]
        :param long_asc_node: longtitude of ascending node [rad]
        :param arg_peri: argument of perihelion [rad]
        :param time_peri: time of perihelion pasage [JD]

        :return: None
        """
        
        self.semi_major_axis = semi_major_axis
        self.eccentricity = eccentricty
        self.inclination = inclination
        self.long_asc_node = long_asc_node
        self.arg_peri = arg_peri
        self.time_peri = time_peri

        self.sun = Horizons(id='@sun', location='399')

    def get_RA_DEC(self, time: np.float64, eq_obliquity: np.float64=0.409092804):
        """
        Calculates and returns the predicted right ascension and declinaiton of the orbital object at a given Julian date

        :param time: time prediction based on [JD]
        :param eq_obliquity: obliquity of the equatorial plane wrt the eclitpic [rad]
        """
        mean_anom = self.K / self.semi_major_axis**(1.5) * (time - self.time_peri)
        ecc_anom = odlib.newton_solve(lambda x : mean_anom - x + self.eccentricity * np.sin(x), lambda x : self.eccentricity * np.cos(x) - 1, mean_anom, 1e-10)
        r_orb = np.array([self.semi_major_axis*(np.cos(ecc_anom) - self.eccentricity), self.semi_major_axis*np.sqrt(1-self.eccentricity**2)*np.sin(ecc_anom), 0])
        r_eq = self._orbital_equatorial_transformation(r_orb, self.arg_peri, self.inclination, self.long_asc_node, eq_obliquity)
        rho = self._get_sun_vector(time) + r_eq
        print(f"sun-earth: {self._get_sun_vector(time)}")

        rho_hat = rho / odlib.magnitude(rho)
        dec = np.arcsin(rho_hat[2])
        asc = odlib.quadrant_checked_angle(rho_hat[0] / np.cos(dec), rho_hat[1] / np.cos(dec))

        return asc, dec

    def _orbital_equatorial_transformation(self, orbital_pos:np.ndarray, arg_peri: np.float64, inclination: np.float64, long_asc_node: np.float64, eclp_obliq: np.float64) -> np.ndarray:
        """
        Transforms position vector from orbital plane to equatorial plane given some orbital elements


        :param orbital_pos: position vector in orbital plane
        :param arg_peri: argument of perihelion of orbit [rad]
        :param inclination: inclinatoin of orbit [rad]
        :param long_asc_node: longtitude of orbit's ascending node [rad]
        :param eclp_obliq: obliquity of 
        """
        cosw = np.cos(arg_peri)
        sinw = np.sin(arg_peri)
        cosi = np.cos(inclination)
        sini = np.sin(inclination)
        coso = np.cos(long_asc_node)
        sino = np.sin(long_asc_node)
        cose = np.cos(eclp_obliq)
        sine = np.sin(eclp_obliq)
        
        return np.array([
            [coso * cosw - cosi * sino * sinw, -cosi * sino * cosw - coso *  sinw, sini * sino],

            [sinw * (cose * cosi * coso - sine * sini) + cose * sino * cosw, cosw * (cose * cosi * coso - sine * sini) - cose * sino * sinw, sine * (-cosi) - cose * sini * coso],

            [sinw * (sine * cosi * coso + cose * sini) + sine * sino * cosw, cosw * (sine * cosi * coso + cose * sini) - sine * sino * sinw, cose * cosi - sine * sini * coso],
        ]) @ orbital_pos
    
    def _get_sun_vector(self, time: np.float64) -> np.ndarray:
        """
        Gets the sun vector from the center of the earth

        :param time: time [JD]
        """
        self.sun.epochs = time
        vecs = self.sun.vectors(refplane="earth")
        return np.array([float(vecs["x"]), float(vecs["y"]), float(vecs["z"])], dtype=np.float64)