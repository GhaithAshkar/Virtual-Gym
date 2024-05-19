import numpy as np

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    a, b, c are tuples representing the coordinates of the points (x, y, z).
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)
