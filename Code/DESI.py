from scipy.interpolate import UnivariateSpline
import numpy as np

BRGArray = np.array([[0.00, 0],
                     [0.05, 1165],
                     [0.15, 3074],
                     [0.25, 1909],
                     [0.35, 732],
                     [0.45, 120],
                     [0.55, 100]])

OtherArray = np.array([
    [0.65, 309, 832, 47],
    [0.75, 2269, 986, 55],
    [0.85, 1923, 662, 61],
    [0.95, 2094, 272, 67],
    [1.05, 1441, 51, 72],
    [1.15, 1353, 17, 76],
    [1.25, 1337, 0, 80],
    [1.35, 523, 0, 83],
    [1.45, 466, 0, 85],
    [1.55, 329, 0, 87],
    [1.65, 126, 0, 87],
    [1.75, 0, 0, 87],
    [1.85, 0, 0, 86],
    [1.9, 0, 0, 0]])

OtherArraySquash = np.zeros((len(OtherArray), 2))
OtherArraySquash[:, 0] = OtherArray[:, 0]
OtherArraySquash[:, 1] = OtherArray[:, 1] + OtherArray[:, 2] + OtherArray[:, 3]

totalArray = np.vstack([BRGArray, OtherArraySquash])
#totalArray = OtherArraySquash
spline = UnivariateSpline(totalArray[:, 0], totalArray[:, 1], s=0, ext=1)

spline = UnivariateSpline(totalArray[:, 0], totalArray[:, 1], s=0, ext=1)
norm = spline.integral(0, 2)
spline_norm = UnivariateSpline(totalArray[:, 0], totalArray[:, 1] / norm, s=0, ext=1)


def DESISpline_normalized(z):
    return spline_norm(z)


def DESISpline(z):
    return spline(z)


def DESIMean():
    return .8
    # return 1.1


def DESIngal():
    return spline.integral(0, 2) / 3600.
