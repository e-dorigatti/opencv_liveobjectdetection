import cv
import cv2
import numpy as np
from sys import argv

def liveDescriptors():
    capture = cv2.VideoCapture(0)
    keypoints, descriptors, knn = initialize('object.jpg')

    while True:
        ret, frame = capture.read()
        if ret == False:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = calculateDescriptors(img)

        for kp in keypoints:
            p = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(frame, p, int(kp.size), (0, 255, 255))
        
        cv.ShowImage('img', cv.fromarray(frame))
        cv.WaitKey(2)

    capture.release()
    cv2.destroyWindow('img')

def compare_images():
    img1 = cv2.imread('object.jpg', cv.CV_LOAD_IMAGE_GRAYSCALE)
    img2 = cv2.imread('test.jpg', cv.CV_LOAD_IMAGE_GRAYSCALE)

    img2 = cv2.resize(img2, (0,0), fx=0.25, fy=0.25)

    keypoints1, descriptors1, knn = initialize(img1)
    print "keypoints for first image: ", len(keypoints1)
    
    keypoints2, descriptors2 = calculateDescriptors(img2)
    print "keypoints for second image: ", len(keypoints2)

    for index, descriptor in enumerate(descriptors2):
        # make it a matrix
        descriptor = np.array(descriptor, np.float32).reshape(1, len(descriptor))
        retval, results, neigh_resp, dists = knn.find_nearest(descriptor, 10)

        for desc, dist in zip(results[0], dists[0]):
            desc = int(desc)

            maxdist = 0.1
            if dist < maxdist:
                c = dist * 255 / maxdist
                (x, y), r = keypoints1[desc].pt, int(keypoints1[desc].size)
                cv2.circle(img1, (int(x), int(y)), r, (c, c, c))

                c = dist * 255 / maxdist
                (x, y), r = keypoints2[index].pt, int(keypoints2[index].size)
                cv2.circle(img2, (int(x), int(y)), r, (c, c, c))

    cv2.namedWindow('img1')
    cv2.imshow('img1', img1)

    cv2.namedWindow('img2')
    cv2.imshow('img2', img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def initialize(image, threshold = None):
    if type(image) == type(''):
        image = cv2.imread(image, cv.CV_LOAD_IMAGE_GRAYSCALE)

    if threshold is not None:
        surf = cv2.SURF(threshold)
    else:
        surf = cv2.SURF()

    kp, descriptors = surf.detectAndCompute(image, None)

    knn = cv2.KNearest()
    knn.train(descriptors, np.array([1] * len(descriptors)), isRegression = False)

    return kp, descriptors, knn

def calculateDescriptors(image, threshold = None):
    if type(image) == type(''):
        image = cv2.imread(image, cv.CV_LOAD_IMAGE_GRAYSCALE)

    if threshold is not None:
        surf = cv2.SURF(threshold)
    else:
        surf = cv2.SURF()

    kp, descriptors = surf.detectAndCompute(image, None)
    return kp, descriptors

if __name__ == '__main__':
    tests = [
        { 'name': 'live',
        'fn': liveDescriptors },

        { 'name': 'compare',
        'fn': compare_images },
    ]
    
    if len(argv) > 1:
        choice = map(lambda test: test['name'], tests).index(argv[1])
        tests[choice]['fn']()
    else:
        print "available choices are ", map(lambda test: test['name'], tests)
