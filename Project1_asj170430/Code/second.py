'''
##############################################################################################################################
#  Written by:- Aadish Joshi
#  Date: March 06, 2019.
#
#
# Problem 2:
# Write a program that gets as input a color image, performs histogram equalization in the Luv domain, and
# writes the scaled image as output. Histogram equalization in Luv is applied to the luminance values, as
# computed in the specified window. It requires a discretization step, where the real-valued L is discretized
# into 101 values.
# As in the first program pixel values outside the window should not be changed. Only pixels within the
# window should be changed.
###############################################################################################################################
'''

import cv2
import numpy as np
import sys

if (len(sys.argv) != 7):
    print(sys.argv[0], ": takes 6 arguments . Not ", len(sys.argv) - 1)
    print(" Expecting arguments : w1 h1 w2 h2 ImageIn ImageOut .")
    print(" Example :", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits .jpg out.png")
    sys.exit()

# take arguments from the user for w1, h1, w2, h2 which is the window size and then the input image on which the
# you want to perform the operation

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]

# checking for edge cases

if (w1 < 0 or h1 < 0 or w2 <= w1 or h2 <= h1 or w2 > 1 or h2 > 1):
    print(" arguments must satisfy 0 <= w1 < w2 <= 1 , 0 <= h1 < h2 <= 1")
    sys.exit()

inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)

if (inputImage is None):
    print(sys.argv[0], ": Failed to read image from : ", name_input)
    sys.exit()

cv2.imshow(" input image : " + name_input, inputImage)

rows, cols, bands = inputImage.shape  # bands == 3


outputImage = np.copy(inputImage)


W1 = int(w1 * (cols - 1))
H1 = int(h1 * (rows - 1))
W2 = int(w2 * (cols - 1))
H2 = int(h2 * (rows - 1))
print("W1 : ", W1, " H1: ", H1, " W2: ", W2, " H2", H2)

# The transformation should be based on the
# historgram of the pixels in the W1 ,W2 ,H1 ,H2 range .
# The following code goes over these pixels

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
#  NonLinear to Linear RGB  Inv Gamma Correlation
###############################################################################################################################
'''

def linearRGB(b, g, r):
    sample = []
    sample.append(b)
    sample.append(g)
    sample.append(r)

    for index in range(0, len(sample)):
        if sample[index] < 0.03928:
            sample[index] = sample[index] / 12.92
        else:
            sample[index] = pow(((sample[index] + 0.055) / 1.055), 2.4)
    return sample[0], sample[1], sample[2]

'''
##############################################################################################################################
#  RGB - XYZ
###############################################################################################################################
'''

def RGB_XYZ(b, g, r):

    X = 0.412453 * r + 0.35758 * g + 0.180423 * b
    Y = 0.212671 * r + 0.71516 * g + 0.072169 * b
    Z = 0.019334 * r + 0.119193 * g + 0.950227 * b
    return X, Y, Z

'''
##############################################################################################################################
#  findL
###############################################################################################################################
'''

def findL(Y):

    if (Y > 0.008856):
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
    uw = ((4 * Xw) / (Xw + (15 * Yw) + (3 * Zw)))
    vw = ((9 * Yw) / (Xw + (15 * Yw) + (3 * Zw)))

    L = findL(Y)

    d = X + (15 * Y) + (3 * Z)
    if (X == 0):
        ud = 0
    else:
        ud = ((4 * X) / d)
    if (Y == 0):
        vd = 0
    else:
        vd = ((9 * Y) / d)
    u = 13 * (L * (ud - uw))
    v = 13 * (L * (vd - vw))
    return L, u, v

'''
##############################################################################################################################
#  Luv_XYZ
###############################################################################################################################
'''

def Luv_XYZ(L, u, v):

    L = float(L)

    Xw = 0.95
    Yw = 1.0
    Zw = 1.09

    uw = ((4 * Xw) / (Xw + (15 * Yw) + (3 * Zw)))
    vw = ((9 * Yw) / (Xw + (15 * Yw) + (3 * Zw)))

    if (L == 0):
        ud = 0
        vd = 0
    else:
        ud = ((u + (13 * uw * L)) / (13 * L))

    if (L != 0):
        vd = ((v + (13 * vw * L)) / (13 * L))

    if (L > 7.9996):
        Y = pow(((L + 16) / 116), 3) * Yw
    else:
        Y = (L / 903.3) * Yw

    if (vd == 0):

        X = 0;
        Z = 0;

    else:
        X = Y * 2.25 * (ud / vd)
        Z = ((Y * (3 - (0.75 * ud) - (5 * vd))) / vd)

    return X, Y, Z

'''
##############################################################################################################################
#  XYZ_RGB
###############################################################################################################################
'''

def XYZ_RGB(X, Y, Z):

    r = 3.240479 * X - 1.53715 * Y - 0.498535 * Z;
    g = -0.969256 * X + 1.875991 * Y + 0.041556 * Z;
    b = 0.055648 * X - 0.204043 * Y + 1.057311 * Z;

    sample = [b, g, r]

    for index in range(0, len(sample)):
        if sample[index] < 0:
            sample[index] = 0
        if sample[index] > 1:
            sample[index] = 1

    return sample[0],  sample[1],  sample[2]

'''
##############################################################################################################################
#  gammaCorrection
###############################################################################################################################
'''

def gamma(b, g, r):

    sample = [b, g, r]

    for i in range(0, len(sample)):
        if (sample[i] < 0.00304):
            sample[i] = 12.92 * sample[i]
        else:
            sample[i] = pow((1.055 * sample[i]), 0.417) - 0.055
    return sample[0], sample[1], sample[2]

'''
##############################################################################################################################
#  Histogram
###############################################################################################################################
'''

L_histogram = []

for i in range(0, 101):
    L_histogram.append(0)

frequency = []
floor = []

Lmin = sys.maxsize
Lmax = -sys.maxsize - 1

for i in range(H1, H2 + 1):
    for j in range(W1, W2 + 1):

        b, g, r = tmp[i, j]

        b, g, r = nonLinearRGB(b, g, r)

        b, g, r = linearRGB(b, g, r)

        X, Y, Z = RGB_XYZ(b, g, r)

        L = findL(Y)
        if (L > Lmax):
            Lmax = L
        if (L < Lmin):
            Lmin = L

        L_histogram[int(L + 0.5)] += 1

frequency.append(L_histogram[0])

# Cumulative frequency

for i in range(1, 101):
    frequency.append(L_histogram[i] + frequency[i - 1])

totalPixels = frequency[100]
rangeOfValues = 101

floor.append((frequency[0] * rangeOfValues) / (2.0 * totalPixels))

for i in range(0, 101):
    floor.append((frequency[i - 1] + frequency[i]) * rangeOfValues / (2 * totalPixels))
    if (floor[i] > 100):
        floor[i] = 100

'''
##############################################################################################################################
#  Output for the second program
###############################################################################################################################
'''

for row in range(H1, H2 + 1):
    for col in range(W1, W2 + 1):
        b, g, r = tmp[row, col]
        b, g, r = nonLinearRGB(b, g, r)

        b, g, r = linearRGB(b, g, r)

        X, Y, Z = RGB_XYZ(b, g, r)
        L, u, v = XYZ_Luv(X, Y, Z)

        if (L > Lmax):
            L = 100
        elif (L < Lmin):
            L = 0
        else:
            L = floor[int(L)]

        X, Y, Z = Luv_XYZ(L, u, v)

        linear_b, linear_g, linear_r = XYZ_RGB(X, Y, Z)

        nonLinear_b, nonLinear_g, nonLinear_r = gamma(linear_b, linear_g, linear_r)

        outputImage[row, col] = [255 * nonLinear_b, 255 * nonLinear_g, 255 * nonLinear_r]

cv2.imshow(" output :", outputImage)
cv2.imwrite(name_output, outputImage)

cv2.waitKey(0)
cv2.destroyAllWindows()