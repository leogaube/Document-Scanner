import cv2 as cv
import imutils
import numpy as np

import os


def find_document_corners(img):
    # 1. preprocessing
    # blur the image to reduce noise
    blurred = cv.GaussianBlur(img, ksize=(5, 5), sigmaX=0)

    # ToDo: more preprocessing

    # 2. find contours
    print("Edge Detection")
    edge_img = cv.Canny(blurred, 75, 200)
    print("Contour Detection")
    contours, hierarchy = cv.findContours(edge_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contour_img = img.copy()
    cv.drawContours(contour_img, contours, -1, (0, 0, 255), 1)

    cv.imshow("Edges", edge_img)
    cv.imshow("Contours", contour_img)

    cv.waitKey(0)
    cv.destroyAllWindows()

    # 3. Find largest contour

    # 4. extract four corner points of largest contour


def perspective_transform(img, edges):
    pass


# Adaptive Thresholding?!
def filter_binary(img):
    pass


def scan(img, save=False):
    doc_corners = find_document_corners(img)
    top_down_img = perspective_transform(img, doc_corners)
    binary_document = filter_binary(top_down_img)

    if save:
        output_filename = os.path.join(".", "results", "document.jpg")
        cv.imwrite(output_filename, binary_document)

    if DEBUG:
        cv.imshow("document", binary_document)
        cv.waitKey()


if __name__ == "__main__":
    DEBUG = False

    filename = os.path.join(".", "imgs", "checkerboard.jpg")
    test_img = cv.imread(filename)
    test_img = imutils.resize(test_img, width=360)

    scan(test_img)
