import cv2
import os


def find_document_edges(img):
    # preprocess

    # smoothing
    # canny vs sobel (computational expensive)
    
    return edges


def perspective_transform(img, edges):
    return result


def filter_binary(img):
    pass


def scan(img, save=False, show=True):
    edges = find_document_edges(img)
    top_down_img = perspective_transform(img, edges)
    binary_document = filter_binary(top_down_img)

    if save:
        output_filename = os.path.join(".", "results", "document.jpg")
        cv2.imwrite(output_filename, binary_document)

    if show:
        cv2.imshow("document", document)
        cv2.waitKey()


if __name__ == "__main__":
    filename = os.path.join(".", "test_imgs", "img1.jpg")
    test_img = cv2.imread(filename)
    scan(test_img)