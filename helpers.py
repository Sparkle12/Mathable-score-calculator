import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

def import_images(path, color):
    files = glob.glob(path)
    return [cv.imread(files[i], cv.IMREAD_COLOR if color else cv.IMREAD_GRAYSCALE) for i in tqdm(range(len(files)), desc="Importing images")]

def sort_and_plot(img):
    plt.plot(np.sort(img.flatten()))
    plt.show()

def histogram(img):
    plt.hist(img.flatten(), 256, [0, 256])
    plt.show()

def right_down_corner(img):
    return img[img.shape[0]//2:, img.shape[1]//2:]

def find_median(img):
    return np.median(img.flatten())

def show_img(img, size = (640, 480)):
    cv.imshow("Image", cv.resize(img, size))
    cv.waitKey(0)
    cv.destroyAllWindows()

def to_binary(img, threshold):
    img[img >= threshold] = 255
    img[img < threshold] = 0
    return img

def subtract_mean(img):
    img = img - np.mean(img)
    img[img < 0] = 0
    return np.array(img, dtype=np.uint8)

def print_where_min(img):
    print(np.where(img == np.min(img)))

def mean_img(images):
    return np.array(np.mean(images, axis=0), dtype= np.uint8)

def standard_deviation(img):
    return np.array(np.std(img, axis = 0), dtype = np.uint8)

def into_chunks(img, chunk_size, number_of_chunks):
    chunks = []
    for i in range(number_of_chunks):
        for j in range(number_of_chunks):
            chunks.append(img[i*chunk_size:(i+1)*chunk_size, j*chunk_size:(j+1)*chunk_size])
    return chunks  