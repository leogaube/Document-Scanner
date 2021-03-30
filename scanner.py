import cv2
import numpy as np

import os


def largest_rectangle(corner_canditates):
    pass


def find_document_edges(img):
    # preprocess

    # smoothing (gaussian blur?) (already part of canny/sobel?)
    # canny vs sobel (computational expensive)

    # Harris corner detection

    # hough transform --> find maxima --> find maxima

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corner_canditates = cv2.cornerHarris(gray, 2, 3, 0.04)

    corners = largest_rectangle(corner_canditates)

    if DEBUG:
        for corner in corners:
            img[corner] = [0, 0, 255]

        cv2.imshow("harris", img)
        cv2.waitKey()

    return corners


def perspective_transform(img, edges):
    pass


def filter_binary(img):
    pass


def scan(img, save=False):
    edges = find_document_edges(img)
    top_down_img = perspective_transform(img, edges)
    binary_document = filter_binary(top_down_img)

    if save:
        output_filename = os.path.join(".", "results", "document.jpg")
        cv2.imwrite(output_filename, binary_document)

    if DEBUG:
        cv2.imshow("document", binary_document)
        cv2.waitKey()


if __name__ == "__main__":
    DEBUG = True

    filename = os.path.join(".", "test_imgs", "checkerboard.jpg")
    test_img = cv2.imread(filename)
    test_img = cv2.resize()

    scan(test_img)
