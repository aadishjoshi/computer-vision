'''
##############################################################################################################################
#  Written by:- Aadish Joshi
#  Date: March 06, 2019.
#
#
# Problem 3:
# This is the same as the first program, except that the scaling is to be performed in the xyY domain. The
# scaling should stretch only the luminance (Y) values. In the specified window perform linear scaling that
# would map the smallest Y value to 0 and the largest Y value to 1.
###############################################################################################################################
'''

import cv2
import numpy as np
import sys

if len(sys.argv) != 7:
    print (sys.argv[0], ': takes 6 arguments . Not ', len(sys.argv) - 1)
    print (' Expecting arguments : w1 h1 w2 h2 ImageIn ImageOut .')
    print (' Example :', sys.argv[0],' 0.2 0.1 0.8 0.5 fruits.jpg out.png')
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]

if w1 < 0 or h1 < 0 or w2 <= w1 or h2 <= h1 or w2 > 1 or h2 > 1:
    print (' arguments must satisfy 0 <= w1 < w2 <= 1 , 0 <= h1 < h2 <= 1')
    sys.exit()

inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)

if inputImage is None:
    print (sys.argv[0], ': Failed to read image from : ', name_input)
    sys.exit()

cv2.imshow(' input image : ' + name_input, inputImage)

(rows, cols, bands) = inputImage.shape  # bands == 3

outputImage = np.copy(inputImage)

W1 = int(w1 * (cols - 1))
H1 = int(h1 * (rows - 1))
W2 = int(w2 * (cols - 1))
H2 = int(h2 * (rows - 1))
print("W1 : ", W1, " H1: ", H1, " W2: ", W2, " H2", H2)


##############################################################################################################################

tmp = np.copy(inputImage)
tmp = tmp.astype(float)

'''
##############################################################################################################################
#  R8-G8-B8 to NonLinear RGB
###############################################################################################################################
'''
def nonLinearRGB(b, g, r):
    b = b / 255
    g = g / 255
    r = r / 255
    return (b, g, r)

'''
##############################################################################################################################
#  NonLinear RGB to Linear ( inv Gamma correlation)
###############################################################################################################################
'''
def LinearRGB(b, g, r):

    sample = []
    sample.append(b)
    sample.append(g)
    sample.append(r)

    for index in range(0, len(sample)):
        if sample[index] < 0.03928:
            sample[index] = sample[index] / 12.92
        else:
            sample[index] = pow((sample[index] + 0.055) / 1.055, 2.4)
    return (sample[0], sample[1], sample[2])

'''
##############################################################################################################################
#  RGB_XYZ
###############################################################################################################################
'''
def RGB_XYZ(b, g, r):
    X = 0.412453 * r + 0.35758 * g + 0.180423 * b
    Y = 0.212671 * r + 0.71516 * g + 0.072169 * b
    Z = 0.019334 * r + 0.119193 * g + 0.950227 * b
    return (X, Y, Z)

'''
##############################################################################################################################
#  XYZ_xyY
###############################################################################################################################
'''
def XYZ_xyY(X, Y, Z):

    if X+Y+Z != 0:
	    x = X / (X + Y + Z)
	    y = Y / (X + Y + Z)
	    return (x, y, Y)
    else:
	    return 0,0,0
'''
##############################################################################################################################
#  xyY_XYZ
###############################################################################################################################
'''
def xyY_XYZ(x, y, Y):

	X=0
	Z=0

	if y!=0:
		X = (x / y) * Y
		Z =(1 - x - y) / y * Y
	return (X, Y, Z)

'''
##############################################################################################################################
#  findL
###############################################################################################################################
'''
def findL(Y):
    if Y > 0.008856:
        L = 116 * pow(Y, 0.33) - 16
    else:
        L = 903.3 * Y
    return L
'''
##############################################################################################################################
#  XYZ_Luv
###############################################################################################################################
'''
def XYZ_Luv(X, Y, Z):

    Xw = 0.95
    Yw = 1.0
    Zw = 1.09
    uw = 4 * Xw / (Xw + 15 * Yw + 3 * Zw)
    vw = 9 * Yw / (Xw + 15 * Yw + 3 * Zw)

    L = findL(Y)

    d = X + 15 * Y + 3 * Z

    if X == 0:  # To handle divide by zero exception
        ud = 0
    else:
        ud = 4 * X / d

    if Y == 0:  # To handle divide by zero exception
        vd = 0
    else:
        vd = 9 * Y / d

    u = 13 * (L * (ud - uw))
    v = 13 * (L * (vd - vw))
    return (L, u, v)

'''
##############################################################################################################################
#  Luv_XYZ
###############################################################################################################################
'''
def Luv_XYZ(L, u, v):

    Xw = 0.95
    Yw = 1.0
    Zw = 1.09

    uw = 4 * Xw / (Xw + 15 * Yw + 3 * Zw)
    vw = 9 * Yw / (Xw + 15 * Yw + 3 * Zw)

    if L == 0:
        ud = 0
        vd = 0
    else:
        ud = (u + 13 * uw * L) / (13 * L)

    if L != 0:
        vd = (v + 13 * vw * L) / (13 * L)

    if L > 7.9996:
        Y = pow((L + 16) / 116, 3) * Yw
    else:
        Y = L / 903.3 * Yw

    if vd == 0:

        X = 0
        Z = 0
    else:

        X = Y * 2.25 * (ud / vd)
        Z = Y * (3 - 0.75 * ud - 5 * vd) / vd
    return (X, Y, Z)
'''
##############################################################################################################################
#  inv_gamma
###############################################################################################################################
'''

def inv_gamma(X, Y, Z):

    r = 3.240479 * X - 1.53715 * Y - 0.498535 * Z
    g = -0.969256 * X + 1.875991 * Y + 0.041556 * Z
    b = 0.055648 * X - 0.204043 * Y + 1.057311 * Z

    sample = [b, g, r]

    for index in range(0, len( sample)):
        if sample[index] < 0:
            sample[index] = 0
        if sample[index] > 1:
            sample[index] = 1

    return (sample[0],sample[1],sample[2])

'''
##############################################################################################################################
#  Gamma
###############################################################################################################################
'''
def Gamma(b, g, r):

    sample = [b, g, r]

    for index in range(0, len(sample)):
        if sample[index] < 0.00304:
            sample[index] = 12.92 * sample[index]
        else:
            sample[index] = pow(1.055 * sample[index], 0.417) - 0.055
    return (sample[0],sample[1],sample[2])


Ymin = 1
Ymax = 0

for i in range(H1, H2 + 1):
    for j in range(W1, W2 + 1):
        (b, g, r) = tmp[i, j]

        (b, g, r) = nonLinearRGB(b, g, r)

        (b, g, r) = LinearRGB(b, g, r)

        (X, Y1, Z) = RGB_XYZ(b, g, r)

        (x, y, Y) = XYZ_xyY(X, Y1, Z)

        if Y > Ymax:
            Ymax = Y
        if Y < Ymin:
            Ymin = Y

'''
##############################################################################################################################
#  Third program output
###############################################################################################################################
'''
for row in range(H1, H2 + 1):
    for col in range(W1, W2 + 1):

        (b, g, r) = tmp[row, col]
        (b, g, r) = nonLinearRGB(b, g, r)
        (b, g, r) = LinearRGB(b, g, r)
        (X, Y, Z) = RGB_XYZ(b, g, r)
        (x, y, Y) = XYZ_xyY(X, Y, Z)

        if Y > Ymax:
            Y = 1

        if Y < Ymin:
            Y = 0
        else:
            Y = (Y - Ymin) * 1 / (Ymax - Ymin)

        (X, Y, Z) = xyY_XYZ(x, y, Y)

        (linear_b, linear_g, linear_r) = inv_gamma(X, Y, Z)

        (nonLinear_b, nonLinear_g, nonLinear_r) = \
            Gamma(linear_b, linear_g, linear_r)

        outputImage[row, col] = [int(255 * nonLinear_b), int(255* nonLinear_g), int(255 * nonLinear_r)]

cv2.imshow(' output :', outputImage)
cv2.imwrite(name_output, outputImage)

cv2.waitKey(0)
cv2.destroyAllWindows()