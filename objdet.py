import cv2
import numpy as np
from sys import argv

class ObjectDetector:
    def __init__(self, obj):
        self.surf = cv2.xfeatures2d.SURF_create()

        self.obj_image = cv2.imread(obj, cv2.COLOR_BGR2GRAY)
        self.obj_image = cv2.resize(self.obj_image, (0, 0), fx=0.05, fy=0.05)
        self.obj_keypoints, self.obj_descriptors = self.surf.detectAndCompute(
            self.obj_image, None
        )

        classes = np.arange(len(self.obj_keypoints), dtype=np.float32)
        self.classifier = cv2.ml.KNearest_create()
        self.classifier.train(self.obj_descriptors, cv2.ml.ROW_SAMPLE, classes)
                              #isRegression=False)

    def detect_object(self, descriptors):
        """
        Detects similar descriptors in the specified black and white image.

        Returns a list of matches (kp1, kp2), where kp1 is the index of the
        keypoint in the object to be matched and kp2 is the index of the
        descriptor.
        """
        good_matches = []
        for index, descriptor in enumerate(descriptors):
            # turn the descriptor into a 1xn matrix
            descriptor = np.array(descriptor, np.float32).reshape(1, len(descriptor))

            # find the most similar descriptor in self.obj_descriptors
            retval, results, neigh_resp, dists = self.classifier.findNearest(
                descriptor, 2
            )

            # i.e. the keypoint1 this descriptor is most similar to
            classification = int(results[0][0])

            if dists[0][0] <= 0.45 * dists[0][1]:
                good_matches.append((classification, index))

        return good_matches

    def draw_frame(self, frame, matches, frame_keypoints):
        """
        Returns an image containing the frame along with the object to be serched
        and lines connecting good matches.

        frame: an image
        matches: list of tuples --> (object_keypoint_index, frame_keypoint_index)
        frame_keypoints: a list of all keypoints found on the frame
        """

        h1, w1 = self.obj_image.shape[:2]
        h2, w2 = frame.shape[:2]

        vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        vis[:h1, :w1] = self.obj_image
        vis[:h2, w1:w1+w2] = frame

        for kp1, kp2 in matches:
            c = (255,255,255)

            (x1, y1), r1 = self.obj_keypoints[kp1].pt, int(self.obj_keypoints[kp1].size)
            (x2, y2), r2 = frame_keypoints[kp2].pt, int(frame_keypoints[kp2].size)
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

            cv2.circle(vis, (x1, y1), 2, c, -1)
            cv2.circle(vis, (x2+w1, y2), 2, c, -1)
            cv2.line(vis, (x1, y1), (x2+w1, y2), c)

        for kp in self.obj_keypoints:
            cv2.circle(vis, (int(kp.pt[0]), int(kp.pt[1])), 2, (100, 100, 100))

        return vis

    def process_frame(self, image):
        """
        Processes the specified frame finding matching descriptors and returning
        and image showing them.
        """
        black_white = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.surf.detectAndCompute(black_white, None)
        matches = self.detect_object(descriptors)
        return self.draw_frame(image, matches, keypoints)

def main():
    capture = cv2.VideoCapture(0)
    detector = ObjectDetector('object.jpg')

    step, i = 2, 0
    while True:
        i += 1

        ret, frame = capture.read()
        assert ret

        frame = frame.copy()[:, ::-1]

        processed = detector.process_frame(frame)
        cv2.imshow('img', processed)

        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break

    capture.release()
    cv2.destroyWindow('img')

if __name__ == '__main__':
    main()
