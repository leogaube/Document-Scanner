import cv2 as cv
import imutils
import numpy as np

import os


def find_document_corners(img):
    # 1. Preprocessing
    # blur the image to reduce noise
    blurred_img = cv.GaussianBlur(img, ksize=(5, 5), sigmaX=0)

    # ToDo: more preprocessing

    # 2. Find all contours
    print("Edge Detection")
    edge_img = cv.Canny(blurred_img, 75, 200)

    print("Contour Detection")
    # use cv.CHAIN_APPROX_SIMPLE --> best case only 4 points needed for perfect rectangular document
    # ToDo: understand retrieval-mode: cv.RETR_EXTERNAL vs RETR_TREE vs RETR_LIST?!
    contours, _hierarchy = cv.findContours(edge_img, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)

    # 3. Find largest contour
    largest_contour = None
    largest_area = -1
    for contour in contours:
        area = cv.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour
    # print(largest_contour)

    # 4. extract four corner points of largest contour
    # (assuming largest contour is our document --> if it fails: Hough Transform?)

    # approximating the contour with (hopefully) only 4 points
    # epsilon 0.01 * contour_perimeter seems to work empirically well for us
    contour_perimeter = cv.arcLength(largest_contour, closed=True)
    doc_corners = cv.approxPolyDP(largest_contour, 0.01 * contour_perimeter, closed=True)

    if DEBUG:
        contour_img = img.copy()
        cv.drawContours(contour_img, contours, -1, (0, 0, 255), 1)
        cv.drawContours(contour_img, doc_corners, -1, (0, 255, 0), 5)

        cv.imshow("Edges", edge_img)
        cv.imshow("Contours", contour_img)

        cv.waitKey(0)
        cv.destroyAllWindows()

    assert len(doc_corners) == 4

    return doc_corners


def perspective_transform(img, doc_corners):
    # ToDo: take rotation into account
    # ToDo: bug top-down image is fliped vertically?! (point order is mixed up?)

    src = np.array([doc_corners[0][0], doc_corners[1][0], doc_corners[2][0], doc_corners[3][0]], dtype="float32")
    print(src)
    dst = np.array([[0, 0], [419, 0], [419, 593], [0, 593]], dtype="float32")
    M = cv.getPerspectiveTransform(src, dst)
    top_down_img = cv.warpPerspective(img, M, (420, 594))

    if DEBUG:
        cv.imshow("Top-Down Document", top_down_img)

        cv.waitKey(0)
        cv.destroyAllWindows()


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

    # if DEBUG:
    #    cv.imshow("document", binary_document)
    #    cv.waitKey()


if __name__ == "__main__":
    DEBUG = True

    filename = os.path.join(".", "imgs", "checkerboard.jpg")
    test_img = cv.imread(filename)
    test_img = imutils.resize(test_img, width=360)

    scan(test_img)
