import cv
import cv2
import numpy as np

def calculateKeyPoints(image):     
    surf = cv2.SURF(3000)
    keyp = surf.detect(image, None)
    return keyp

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
    keyp = calculateKeyPoints(img)
    print len(keyp), ' found'
    
    showKeyPoints(img, keyp)

#capture = cv.CaptureFromCAM(0)
#count = 0
#while count < 250:
#    image = cv.QueryFrame(capture)
#    cv.ShowImage('Image Window', image)
#    cv.WaitKey(2)
#    count += 1
