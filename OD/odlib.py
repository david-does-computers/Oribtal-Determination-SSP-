import numpy as np 
from astropy.time import Time


def HMS2deg(hours: np.int8, minutes:np.int8 , seconds:np.float64, radians=False) -> np.float64: #Defaults to degrees if not specified
    """
    Converts hexagesimal right ascension to decimal degrees
    """
    decimal_deg = hours * 15 #15 degrees in one hour
    decimal_deg += minutes * 15/60 #Continue the sum with the relevant conversions
    decimal_deg += seconds * 15/3600
    decimal_deg %= 360
    if not radians: #If radians are not desired, return RA in degrees
        return decimal_deg #Return will break the code block and so radians won't be returned
    return (decimal_deg * np.pi / 180) #Return the RA converted to radians; wont'e be reached unless above conditoin is met

def deg2HMS(degrees: float) -> tuple[np.int8, np.int8, np.float64]:
    """
    Converts decimal degrees right ascension to hexagesimal 
    """
    hours = (degrees // 15) % 24
    mins = (degrees % 15) * 60 / 15
    secs = (mins % 1)* 60
    return hours, np.floor(mins), secs

def DMS2deg(degrees:float, arcminutes:float, arcseconds:float) -> np.float64:
    """
    Converts hexagesimal declination to decimal degrees
    """
    return  np.copysign(np.abs(degrees) + arcminutes/60 + arcseconds/3600, degrees) #multiply by the sign of degrees to ensure the - distributes

def deg2DMS(degrees: float) -> tuple[np.int8, np.int8, np.float64]:
    """
    Converts decimal degrees declination to hexagesimal
    """
    arcmins = np.abs(degrees) % 1 * 60 #Work with absolute value to ensure correct arcmins and arcsecs
    arcsecs = (arcmins % 1) * 60
    return np.sign(degrees)*float(int(np.abs(degrees))), np.floor(arcmins), arcsecs #Multiply by the sign of degrees in the end

def date2JD(datetime: str) -> np.float64:
    """
    Returns the Julian Date of a date

    :param datetime: Valid datetime string
    """
    return Time(datetime).jd


# Mathematics

def quadrant_checked_angle(cos_x:float, sin_x: float) -> np.float64:
    """
    Returns the quadrant checked angle (in radians) given the cosine of the angle and a positive multiple of its sine value
    """
    return np.copysign(np.arccos(cos_x), sin_x) % (2*np.pi) #If sin < 0, then theta is the negative of what it would have been with sin > 0 (the default of arccos)

def newton_solve(f, f_prime, guess:np.float64, tolerance:np.float64) -> np.float64:
    """
    Newton-Raphson approxiamtion to the zero of a function given the function, its derivative, an inital guess, and an error tolerance in the computed value 
    """
    if (np.abs(f(guess)) > np.abs(f_prime(guess)) * tolerance) and f_prime(guess):
        return newton_solve(f, f_prime, guess-f(guess)/f_prime(guess), tolerance)
    else:
        return guess
    
def magnitude(vector: np.ndarray) -> np.float64:
    """
    Reutrns the magniutde of an inclination-d vector
    """
    return np.sqrt(vector.dot(vector))

def magnitude_sq(vector: np.ndarray) -> np.float64:
    """
    Reutrns the square of the magniutde of an inclination-d vector
    """
    return vector.dot(vector)

def triple(A:np.ndarray, B:np.ndarray, C:np.ndarray) -> np.ndarray:
    """
    Returns the triple product: A dot (B cross C)
    """
    return A.dot(np.cross(B, C))

def relative_difference(test_value:np.float64, real_value:np.float64) -> np.float64:
    return abs(test_value-real_value)/real_value