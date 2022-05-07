import cv2 as cv
import imutils
import numpy as np

import os
import sys


def find_document_corners(img):
    # 1. Preprocessing
    # Size limit
    (h, w) = img.shape[:2]
    scale = 1.0
    if h > 1000 or w > 1000:
        if h > w:
            scale *= h / 1000.0
            img = imutils.resize(img, height=1000)
        else:
            scale *= w / 1000.0
            img = imutils.resize(img, width=1000)

    # blur the image to reduce noise
    blurred_img = cv.GaussianBlur(img, ksize=(5, 5), sigmaX=0)

    # 2. Find all contours
    # print("Edge Detection")
    edge_img = cv.Canny(blurred_img, 160, 210)

    # print("Contour Detection")
    # use cv.CHAIN_APPROX_SIMPLE --> best case only 4 points needed for perfect rectangular document
    # ToDo: understand retrieval-mode: cv.RETR_EXTERNAL vs RETR_TREE vs RETR_LIST?!
    contours, _hierarchy = cv.findContours(edge_img, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)

    contour_img = img.copy()
    # 3. Find largest contour which can be approximated using only four points

    largest_contour = None
    largest_area = -1
    for contour in contours:
        perimeter = cv.arcLength(contour, closed=True)
        low_poly_contour = cv.approxPolyDP(contour, 0.02 * perimeter, closed=True)

        if len(low_poly_contour) == 4:
            (x, y, w, h) = cv.boundingRect(contour)
            # cv.rectangle(contour_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            area = w * h
            if area > largest_area and area > 10000:
                largest_area = area
                largest_contour = low_poly_contour

    if largest_contour is None:
        return None

    # 4. extract four corner points of largest contour
    # (assuming largest contour is our document --> if it fails: Hough Transform?)

    if DEBUG:
        cv.drawContours(contour_img, contours, -1, (0, 0, 255), 1)
        cv.drawContours(contour_img, largest_contour, -1, (0, 255, 0), 5)

        cv.namedWindow("Edges", cv.WINDOW_NORMAL)
        cv.namedWindow("Contours", cv.WINDOW_NORMAL)

        cv.imshow("Edges", edge_img)
        cv.imshow("Contours", contour_img)

        cv.resizeWindow("Edges", 707, 1000)
        cv.resizeWindow("Contours", 707, 1000)

        cv.waitKey(0)
        cv.destroyAllWindows()

    doc_corners = largest_contour * scale
    assert len(doc_corners) == 4

    return doc_corners


def perspective_transform(img, doc_corners):
    # print("Persective Transformation")
    # ToDo: take rotation into account

    # Final Image dimensions (keep aspect ratio of DIN A4)!
    WIDTH = 210 * 5
    HEIGHT = 297 * 5

    src = np.array([doc_corners[1][0], doc_corners[2][0], doc_corners[0][0], doc_corners[3][0]], dtype="float32")
    dst = np.array([[0, 0], [0, HEIGHT - 1], [WIDTH - 1, 0], [WIDTH - 1, HEIGHT - 1]], dtype="float32")
    M = cv.getPerspectiveTransform(src, dst)
    top_down_img = cv.warpPerspective(img, M, (WIDTH, HEIGHT))

    if DEBUG:
        cv.imshow("Top-Down Document", top_down_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return top_down_img


def filter_binary(img):
    # print("Adaptive Thresholding")
    # Adaptive Thresholding
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    binary_img = cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 5)

    if DEBUG:
        cv.imshow("Final Document", binary_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return binary_img


if __name__ == "__main__":
    DEBUG = False
    SAVE = True

    test_folder = os.path.join(".", "imgs", "test")
    for image_name in os.listdir(test_folder):
        filename = os.path.join(test_folder, image_name)
        if not os.path.isfile(filename):
            continue
        img = cv.imread(filename)

        print(f"\ntrying to scan document: {filename}")
        doc_corners = find_document_corners(img)
        if doc_corners is None:
            print("FAILURE: unable to find a countour that can be approximated using 4 corner points!")
            continue
        top_down_img = perspective_transform(img, doc_corners)
        binary_document = filter_binary(top_down_img)

        if SAVE:
            result_dir = os.path.join(".", "imgs", "results")
            if not os.path.isdir(result_dir):
                os.mkdir(result_dir)
            output_filename = os.path.join(result_dir, image_name)
            cv.imwrite(output_filename, binary_document)
            print(f"SUCCESS: document has been saved to: {output_filename}")
