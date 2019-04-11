'''
Written By: Aadish Joshi
Date: April 2nd, 2019

Write an OpenCV program that can detect a winking face. You may want to build your program
by changing the example program DetectWink.py.


commands:
python DetectWink1.py newimages output
python DetectWink1.py

'''

'''
#####################################################################################
Imports
#####################################################################################
'''
from os import listdir, makedirs
from os.path import isfile, join, exists
import numpy as np
from fileinput import filename
import itertools
import cv2
import sys
import copy
'''
#####################################################################################
PreTrained Cascades
#####################################################################################
'''
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                     + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                    + 'haarcascade_eye.xml')
glass_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                      + 'haarcascade_eye_tree_eyeglasses.xml')
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                         + 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                          + 'haarcascade_righteye_2splits.xml')

'''
#####################################################################################
Draw Rectangle
#####################################################################################
'''
def draw_rect(frame, x,y,h,w, color):
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

'''
#####################################################################################
Detect Wink
#####################################################################################
'''
def detectWink(frame, location, ROI):

    #Detect eye
    eyes = eye_cascade.detectMultiScale(
        ROI, 1.1, 9, 0 | cv2.CASCADE_SCALE_IMAGE, (10, 20))

    #if same eye has been detected many times, we merge all the overlays.
    eyes = OverlayMerge(eyes)

    # If more than 2 eyes are detected
    if len(eyes) > 2:
        average = sum(eyes)/len(eyes)
        up = []
        low = []
        right = []
        left = []
        for e in eyes:
            if e[1] < average[1]:
                up += [e]
            else:
                low += [e]
            if e[0] < average[0]:
                left += [e]
            else:
                right += [e]
            x, y, w, h = e[0]+location[0], e[1]+location[1], e[2], e[3]
            draw_rect(frame, x, y, h, w, (0, 0, 255))
        if len(right) == 2:
            eyes = right        
        elif len(up) == 2:
            eyes = up
        elif len(left) == 2:
            eyes = left
        elif len(low) == 2:
            eyes = low

    # if less than 1 eye is detected, we check the glass cascade
    if len(eyes) < 1:
        ROI = ROI[0:int(len(ROI)*4/5), 0:len(ROI[0])]
        eyes = glass_cascade.detectMultiScale(ROI, 1.15, 3,
               0 | cv2.CASCADE_SCALE_IMAGE, (5, 10))
        if len(eyes) < 1:
            eyes = eye_cascade.detectMultiScale(
                ROI, 1.05, 3, 0 | cv2.CASCADE_SCALE_IMAGE, (5, 10))
        eyes = OverlayMerge(eyes)
        for e in eyes:
            x, y, w, h = e[0]+location[0], e[1]+location[1], e[2], e[3]
            draw_rect(frame, x, y, h, w, (0, 0, 255))
        if len(eyes) == 1:
            return True
    
    # if 1 eye is detected. i.e. a perfect wink
    if len(eyes) == 1:
        x, y, w, h = eyes[0][0], eyes[0][1], eyes[0][2], eyes[0][3]
        x += location[0]
        y += location[1]
        draw_rect(frame, x, y, h, w, (0, 0, 255))
        return True


    # if 2 eyes are detected
    if len(eyes) == 2:
        
        # check if left eye is closed and right eye is opened.
        LE = None
        LX = 0
        for e in eyes:
            if LX < e[0]:
                LX = e[0]
                LE = e
        x, y, w, h = LE[0], LE[1], LE[2], LE[3]
        L_ROI = ROI[y-10:y+h+10, x-10:x+w+10]
        LE = left_eye_cascade.detectMultiScale(L_ROI, 1.15, 7,
                   0 | cv2.CASCADE_SCALE_IMAGE, (10, 20))
        if len(LE) < 1:
            LE = right_eye_cascade.detectMultiScale(L_ROI, 1.15, 7,
                       0 | cv2.CASCADE_SCALE_IMAGE, (10, 20))
        x += location[0]
        y += location[1]
        if len(LE) > 0:
            draw_rect(frame, x, y, h, w, (0, 0, 255))
        

        # check if right eye is closed an left eye is opened
        RE = None
        RX = len(ROI[0])
        for e in eyes:
            if RX > e[0]:
                RX = e[0]
                RE = e
        x, y, w, h = RE[0], RE[1], RE[2], RE[3]
        R_ROI = ROI[y-10:y+h+10, x-10:x+w+10]
        RE = right_eye_cascade.detectMultiScale(R_ROI, 1.15, 7,
                    0 | cv2.CASCADE_SCALE_IMAGE, (10, 20))
        if len(RE) < 1:
            RE = left_eye_cascade.detectMultiScale(R_ROI, 1.15, 7,
                        0 | cv2.CASCADE_SCALE_IMAGE, (10, 20))
        x += location[0]
        y += location[1]
        if len(RE) > 0:
            draw_rect(frame, x, y, h, w, (0, 0, 255))
        
        # if both eyes are detected and they addup to 1 then it is a perfect wink
        if len(LE) + len(RE) == 1:
            return True
    
    return False

'''
#####################################################################################
detect method for face detection
#####################################################################################
'''
def detect(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scaleFactor = 1.1
    minNeighbors = 20
    flag = 0 | cv2.CASCADE_SCALE_IMAGE
    minSize = (30, 30)
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor,
        minNeighbors,
        flag,
        minSize)
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor,
            5,
            flag,
            minSize)
    detected = 0
    not_detected = 0
    faces = OverlayMerge(faces)
    for f in faces:
        x, y, w, h = f[0], f[1], f[2], f[3]
        faceROI = gray_frame[y:y+h, x:x+w]
        if detectWink(frame, (x, y), faceROI):
            detected += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            not_detected += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return detected, not_detected

'''
#####################################################################################
Overlay merge: if one eye is detected more than once, then it should be merged 
into single unit.
#####################################################################################
'''
def OverlayMerge(Entity):
    overlay = True
    temp = []
    while True:
        if len(Entity) <= 1:
            break
        for a, b in list(itertools.product(Entity, Entity)):
            if np.array_equal(a, b):
                continue
            else:
                if ifoverlay(a[0], a[1], a[0]+a[2], a[1]+a[3], b[0],
                                 b[1], b[0]+b[2], b[1]+b[3]):
                    temp = [x for x in Entity if not
                                   (np.array_equal(b, x) or
                                    np.array_equal(a, x))]
                    temp += [[min(a[0], b[0]), min(a[1], b[1]), max(a[0]
                                     + a[2], b[0]+b[2])-min(a[0], b[0]), max(
                                     a[1]+a[3], b[1]+b[3])-min(a[1], b[1])]]
                    Entity = temp
                    break
            overlay = False
        if not overlay:
            break
    return Entity

'''
#####################################################################################
ifoverlay: to check if there is an overlap
#####################################################################################
'''
def ifoverlay(Lx1, Ly1, Rx1, Ry1, Lx2, Ly2, Rx2, Ry2):
    if bool(Ly1 < Ry2) ^ bool(Ly2 < Ry1): # one on the top of others
        return False
    if bool(Lx1 > Rx2) ^ bool(Lx2 > Rx1): #total left
        return False
    return True

'''
#####################################################################################
run_on_folder Method
#####################################################################################
'''

def run_on_folder(folder, output):
    if(folder[-1] != "/"):
        folder = folder + "/"
    files = [join(folder, f) for f in listdir(folder)
             if isfile(join(folder, f))]
    if not exists(output+"/"):
        makedirs(output+"/")
    print("Total images: ", len(files))
    totalCount = 0
    total_non_detect = 0
    for f in files:
        img = cv2.imread(f, 1)
        if type(img) is np.ndarray:
            img_arr = [copy.deepcopy(img), copy.deepcopy(img), copy.deepcopy(img)] 
            wink = [0, 0, 0]
            filename = f.split("/")[-1] 
            for option in [0, 1]:
                wink[option],non_detected = detect(img_arr[option])
            cnt = max(set(wink), key=wink.count)
            img_out = img_arr[wink.index(cnt)]
            cv2.imwrite(output+"/"+filename, img_out)
            totalCount += cnt
            total_non_detect += non_detected
    return totalCount, total_non_detect

'''
#####################################################################################
runonVideo
#####################################################################################
'''
def runonVideo():
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showlive = True
    while(showlive):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            exit()

        detect(frame)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showlive = False

    videocapture.release()
    cv2.destroyAllWindows()

'''
#####################################################################################
Main method
#####################################################################################
'''
if __name__ == "__main__":
    if len(sys.argv) != 1 and len(sys.argv) != 3:
        print(sys.argv[0] + ": got " + str(len(sys.argv) - 1)
              + "arguments. Expecting 0 or 2:[image-folder] [output-folder]")
        exit()

    if(len(sys.argv) == 3): 
        folderName = sys.argv[1]
        outputFolder = sys.argv[2]
        detections,non_detection = run_on_folder(folderName, outputFolder)
        print("Total of ", detections, "detections")
        print("Total of ", non_detection, "non_detection")
    else: # no arguments
        runonVideo()