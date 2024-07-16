import numpy as np
import odlib

class OrbitDetermination:

    GAUSSIAN_DAY = 182.62844915 / np.pi      # Solar days per Gaussian Day
    K = 0.0172020989484                      # Gaussian gravitational constant

    def __init__(self, r: np.ndarray[np.float64], rdot:np.ndarray[np.float64], time0:np.float64) -> None:
        """
        Initializes the Orbit object using the position and velocity vectors as well as the corresponding Julian date (time)

        :param r: position vector [AU]
        :param rdot: velocity vector [AU / GauDay]
        :param time0: time corresponding to position and velocity [Julian Days]

        :return: None
        """
        
        self.r = r
        self.rdot = rdot
        self.t0 = time0
    
    def calc_orbital_elements(self) -> None:
        self._calc_spec_ang_mom()
        self._calc_semi_major_axis()
        self._calc_eccentricity()
        self._calc_inclination()
        self._calc_long_asc_node()
        self._calc_arg_peri()
        self._calc_anom()
        self._calc_time_peri()

    def get_orbital_elements(self) -> dict[np.float64]:
        return (
            self.semi_major_axis, 
            self.eccentricity, 
            self.inclination, 
            self.long_asc_node, 
            self.arg_peri, 
            self.time_peri
        )

    def get_orbital_elements_dict(self) -> dict[np.float64]:
        return {
            "a": self.semi_major_axis, 
            "e": self.eccentricity, 
            "i": np.degrees(self.inclination), 
            "o": np.degrees(self.long_asc_node), 
            "w": np.degrees(self.arg_peri), 
            "T": self.time_peri
            }
        

    def _calc_spec_ang_mom(self) -> None:
        """
        Calculates the specific angular momentum vector [AU^2 / GaussDay]
        """
        self.spec_ang_mom = np.cross(self.r, self.rdot)

    def _calc_semi_major_axis(self) -> None:
        """
        Calculates the semi-major axis [AU]
        """
        self.semi_major_axis = (2 / odlib.magnitude(self.r) - odlib.magnitude_sq(self.rdot)) ** (-1)

    def _calc_eccentricity(self) -> None:
        """
        Calculates the eccentricity
        """
        self.eccentricity = np.sqrt(1 - odlib.magnitude_sq(self.spec_ang_mom) / self.semi_major_axis)

    def _calc_inclination(self) -> None:
        """
        Calculates the inclination [radians]
        """
        self.inclination = np.arccos(self.spec_ang_mom[2] / odlib.magnitude(self.spec_ang_mom))

    def _calc_long_asc_node(self) -> None:
        """
        Calculautes the longititude of ascending node [radians]
        """
        # qaudrant checked arrival at Omega
        self.long_asc_node = odlib.quadrant_checked_angle(
            -self.spec_ang_mom[1] / odlib.magnitude(self.spec_ang_mom) / np.sin(self.inclination),
            self.spec_ang_mom[0] / np.sin(self.inclination)
        )
        
    def _calc_arg_peri(self) -> None:
        """
        Calculautes the longititude of ascending node [radians]
        """

        # Caclualte the angle from the line of the ascending node to the asteroid (quadrant checked)
        U = odlib.quadrant_checked_angle(
            (self.r[0] * np.cos(self.long_asc_node) + self.r[1] * np.sin(self.long_asc_node)) / odlib.magnitude(self.r), 
            self.r[2] / np.sin(self.inclination)
        )

        # Calculate the angle of the asteroid from the periapsis (quadrant checked)
        self.nu = odlib.quadrant_checked_angle(
            (self.semi_major_axis*(1-self.eccentricity**2) / odlib.magnitude(self.r) - 1) / self.eccentricity, 
            self.r.dot(self.rdot)
        )
        
        self.arg_peri = (U - self.nu) % (2*np.pi)

    def _calc_anom(self) -> None:
        """
        Calculaute the eccentric and mean anomalies
        """
        self.ecc_anom = odlib.quadrant_checked_angle(
            self.eccentricity + odlib.magnitude(self.r)* np.cos(self.nu) / self.semi_major_axis,
            np.sin(self.nu)
        )

        self.mean_anom = self.ecc_anom - self.eccentricity * np.sin(self.ecc_anom)


    def _calc_time_peri(self) -> None:
        """
        Caclulate the time of perhileion passage (Julan Days)
        """
        self.time_peri = self.t0 - self.mean_anom * self.semi_major_axis**(1.5) / self.K


