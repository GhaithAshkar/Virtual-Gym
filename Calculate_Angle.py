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

    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees
