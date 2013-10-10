import cv
import cv2
import numpy as np

def liveDescriptors():
    capture = cv2.VideoCapture(0)
    count = 0
    while count < 250:
        ret, frame = capture.read()
        if ret == False:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, d = calculateDescriptors(img)

        for kp in keypoints:
            p = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(frame, p, int(kp.size), (0, 255, 255))
        
        cv.ShowImage('img', cv.fromarray(frame))
        cv.WaitKey(2)
        
        count += 1

    capture.release()
    cv2.destroyWindow('img')

def calculateDescriptors(image, threshold = 3000):
    surf = cv2.SURF(threshold)
    kp, descriptors = surf.detectAndCompute(image, None)
    return kp, descriptors

def showKeyPoints(image, keypoints):
    window = cv2.namedWindow('image')

    for kp in keypoints:
        p = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(img, p, int(kp.size), (0, 255, 255))

    image = cv2.resize(image, (800, 600), interpolation=cv.CV_INTER_AREA)
    cv2.imshow('image', image)
    cv.ResizeWindow('image', 800, 600)
    cv.WaitKey()
    
    cv2.destroyWindow('image')

if __name__ == '__main__':
    img = cv2.imread('object.jpg', cv.CV_LOAD_IMAGE_GRAYSCALE)

    print 'looking for keypoints...'
    keyp, descr = calculateDescriptors(img)
    print len(keyp), ' found'
    
    # showKeyPoints(img, keyp)
    liveDescriptors()
