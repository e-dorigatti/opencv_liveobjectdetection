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
    imag1 = cv2.imread('object.jpg')
    imag2 = cv2.imread('test.jpg')

    img1 = cv2.cvtColor(imag1, cv.CV_BGR2GRAY)
    img2 = cv2.cvtColor(imag2, cv.CV_BGR2GRAY)

    img1 = cv2.resize(img1, (0,0), fx=0.25, fy=0.25)
    img2 = cv2.resize(img2, (0,0), fx=0.25, fy=0.25)

    keypoints1, descriptors1, knn = initialize(img1)
    keypoints2, descriptors2 = calculateDescriptors(img2)

    print "keypoints for first image: ", len(keypoints1)
    print "keypoints for second image: ", len(keypoints2)

    good_matches = []   # list of tuples --> (keypoint1, keypoint2)
    for index, descriptor in enumerate(descriptors2):
        # turn the descriptor into a 1xn matrix
        descriptor = np.array(descriptor, np.float32).reshape(1, len(descriptor))
        retval, results, neigh_resp, dists = knn.find_nearest(descriptor, 2)

        # i.e. the keypoint1 this descriptor is most similar to
        classification = int(results[0][0])

        if dists[0][0] <= 0.50 * dists[0][1]:
            good_matches.append((classification, index))

    # output image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for kp1, kp2 in good_matches:
        c = (255,255,255)

        (x1, y1), r1 = keypoints1[kp1].pt, int(keypoints1[kp1].size)
        cv2.circle(vis, (int(x1), int(y1)), 3, c)

        (x2, y2), r2 = keypoints2[kp2].pt, int(keypoints2[kp1].size)
        cv2.circle(vis, (int(x2+w1), int(y2)), 3, c)

        cv2.line(vis, (int(x1),int(y1)), (int(x2+w1),int(y2)), c)

    cv2.imshow('vis', vis)
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
    classes = np.arange(len(kp), dtype = np.float32)

    knn = cv2.KNearest()
    knn.train(descriptors, classes, isRegression = False)

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
