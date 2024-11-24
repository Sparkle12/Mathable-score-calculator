from copy import deepcopy
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from helpers import *
from tqdm import tqdm
import math
EMPTY = -1

ANY = 0
PLUS = 1
MINUS = 2
MULTIPLY = 3
DIVIDE = 4

piece_board = [
                [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
                [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
                [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
                [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
                [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
                [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
                [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, 1, 2, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
                [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, 3, 4, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
                [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
                [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
                [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
                [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
                [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
                [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
              ]

multiplier_board = [
                    [3, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 3],
                    [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
                    [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1],
                    [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
                    [1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
                    [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1],
                    [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
                    [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1],
                    [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
                    [3, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 3]
                   ]

sign_board = [
                [ANY, ANY, ANY, ANY, ANY, ANY, ANY, ANY, ANY, ANY, ANY, ANY, ANY, ANY],
                [ANY, ANY, ANY, ANY, DIVIDE, ANY, ANY, ANY, ANY, DIVIDE, ANY, ANY, ANY, ANY],
                [ANY, ANY, ANY, ANY, ANY, MINUS, ANY, ANY, MINUS, ANY, ANY, ANY, ANY, ANY],
                [ANY, ANY, ANY, ANY, ANY, ANY, PLUS, MULTIPLY, ANY, ANY, ANY, ANY, ANY, ANY],
                [ANY, DIVIDE, ANY, ANY, ANY, ANY, MULTIPLY, PLUS, ANY, ANY, ANY, ANY, DIVIDE, ANY],
                [ANY, ANY, MINUS, ANY, ANY, ANY, ANY, ANY, ANY, ANY, ANY, MINUS, ANY, ANY],
                [ANY, ANY, ANY, MULTIPLY, PLUS, ANY, ANY, ANY, ANY, MULTIPLY, PLUS, ANY, ANY, ANY],
                [ANY, ANY, ANY, PLUS, MULTIPLY, ANY, ANY, ANY, ANY, PLUS, MULTIPLY, ANY, ANY, ANY],
                [ANY, ANY, MINUS, ANY, ANY, ANY, ANY, ANY, ANY, ANY, ANY, MINUS, ANY, ANY],
                [ANY, DIVIDE, ANY, ANY, ANY, ANY, PLUS, MULTIPLY, ANY, ANY, ANY, ANY, DIVIDE, ANY],
                [ANY, ANY, ANY, ANY, ANY, ANY, MULTIPLY, PLUS, ANY, ANY, ANY, ANY, ANY, ANY],
                [ANY, ANY, ANY, ANY, ANY, MINUS, ANY, ANY, MINUS, ANY, ANY, ANY, ANY, ANY],
                [ANY, ANY, ANY, ANY, DIVIDE, ANY, ANY, ANY, ANY, DIVIDE, ANY, ANY, ANY, ANY],
                [ANY, ANY, ANY, ANY, ANY, ANY, ANY, ANY, ANY, ANY, ANY, ANY, ANY, ANY],
             ]

def get_possible_templates(line, col):
    i = line - 1
    j = col - 1
    posible = []
    if sign_board[i][j] == ANY:
        if i >= 2:
            if piece_board[i - 1][j] != EMPTY and piece_board[i - 2][j] != EMPTY:
                posible.append(piece_board[i - 1][j] + piece_board[i - 2][j])
                posible.append(piece_board[i - 1][j] * piece_board[i - 2][j])
                posible.append(abs(piece_board[i - 1][j] - piece_board[i - 2][j]))
                if piece_board[i - 1][j] / piece_board[i - 2][j] == piece_board[i - 1][j] // piece_board[i - 2][j]:
                    posible.append(piece_board[i-1][j] // piece_board[i-2][j])
                elif piece_board[i - 2][j] / piece_board[i - 1][j] == piece_board[i - 2][j] // piece_board[i - 1][j]:
                    posible.append(piece_board[i - 2][j] // piece_board[i - 1][j])
        if i <= 11:
            if piece_board[i + 1][j] != EMPTY and piece_board[i + 2][j] != EMPTY:
                posible.append(piece_board[i + 1][j] + piece_board[i + 2][j])
                posible.append(piece_board[i + 1][j] * piece_board[i + 2][j])
                posible.append(abs(piece_board[i + 1][j] - piece_board[i + 2][j]))
                if piece_board[i + 1][j] / piece_board[i + 2][j] == piece_board[i + 1][j] // piece_board[i + 2][j]:
                    posible.append(piece_board[i + 1][j] // piece_board[i + 2][j])
                elif piece_board[i + 2][j] / piece_board[i + 1][j] == piece_board[i + 2][j] // piece_board[i + 1][j]:
                    posible.append(piece_board[i + 2][j] // piece_board[i + 1][j])
        if j >= 2:
            if piece_board[i][j - 1] != EMPTY and piece_board[i][j - 2] != EMPTY:
                posible.append(piece_board[i][j - 1] + piece_board[i][j - 2])
                posible.append(piece_board[i][j - 1] * piece_board[i][j - 2])
                posible.append(abs(piece_board[i][j - 1] - piece_board[i][j - 2]))
                if piece_board[i][j - 1] / piece_board[i][j - 2] == piece_board[i][j - 1] // piece_board[i][j - 2]:
                    posible.append(piece_board[i][j - 1] // piece_board[i][j - 2])
                elif piece_board[i][j - 2] / piece_board[i][j - 1] == piece_board[i][j - 2] // piece_board[i][j - 1]:
                    posible.append(piece_board[i][j - 2] // piece_board[i][j - 1])
        if j <= 11:
            if piece_board[i][j + 1] != EMPTY and piece_board[i][j + 2] != EMPTY:
                posible.append(piece_board[i][j + 1] + piece_board[i][j + 2])
                posible.append(piece_board[i][j + 1] * piece_board[i][j + 2])
                posible.append(abs(piece_board[i][j + 1] - piece_board[i][j + 2]))
                if piece_board[i][j + 1] / piece_board[i][j + 2] == piece_board[i][j + 1] // piece_board[i][j + 2]:
                    posible.append(piece_board[i][j + 1] // piece_board[i][j + 2])
                elif piece_board[i][j + 2] / piece_board[i][j + 1] == piece_board[i][j + 2] // piece_board[i][j + 1]:
                    posible.append(piece_board[i][j + 2] // piece_board[i][j + 1])
    elif sign_board[i][j] == MULTIPLY:
        if i >= 2:
            if piece_board[i - 1][j] != EMPTY and piece_board[i - 2][j] != EMPTY:
                posible.append(piece_board[i - 1][j] * piece_board[i - 2][j])
        if i <= 11:
            if piece_board[i + 1][j] != EMPTY and piece_board[i + 2][j] != EMPTY:
                posible.append(piece_board[i + 1][j] * piece_board[i + 2][j])
        if j >= 2:
            if piece_board[i][j - 1] != EMPTY and piece_board[i][j - 2] != EMPTY:
                posible.append(piece_board[i][j - 1] * piece_board[i][j - 2])
        if j <= 11:
            if piece_board[i][j + 1] != EMPTY and piece_board[i][j + 2] != EMPTY:
                posible.append(piece_board[i][j + 1] * piece_board[i][j + 2])
    elif sign_board[i][j] == PLUS:
        if i >= 2:
            if piece_board[i - 1][j] != EMPTY and piece_board[i - 2][j] != EMPTY:
                posible.append(piece_board[i - 1][j] + piece_board[i - 2][j])
        if i <= 11:
            if piece_board[i + 1][j] != EMPTY and piece_board[i + 2][j] != EMPTY:
                posible.append(piece_board[i + 1][j] + piece_board[i + 2][j])
        if j >= 2:
            if piece_board[i][j - 1] != EMPTY and piece_board[i][j - 2] != EMPTY:
                posible.append(piece_board[i][j - 1] + piece_board[i][j - 2])
        if j <= 11:
            if piece_board[i][j + 1] != EMPTY and piece_board[i][j + 2] != EMPTY:
                posible.append(piece_board[i][j + 1] + piece_board[i][j + 2])
    elif sign_board[i][j] == MINUS:
        if i >= 2:
            if piece_board[i - 1][j] != EMPTY and piece_board[i - 2][j] != EMPTY:
                posible.append(abs(piece_board[i - 1][j] - piece_board[i - 2][j]))
        if i <= 11:
            if piece_board[i + 1][j] != EMPTY and piece_board[i + 2][j] != EMPTY:
                posible.append(abs(piece_board[i + 1][j] - piece_board[i + 2][j]))
        if j >= 2:
            if piece_board[i][j - 1] != EMPTY and piece_board[i][j - 2] != EMPTY:
                posible.append(abs(piece_board[i][j - 1] - piece_board[i][j - 2]))
        if j <= 11:
            if piece_board[i][j + 1] != EMPTY and piece_board[i][j + 2] != EMPTY:
                posible.append(abs(piece_board[i][j + 1] + piece_board[i][j + 2]))          
    return set(posible)
def get_board(img):

    copy_img = deepcopy(img)
    copy_img = cv.cvtColor(copy_img, cv.COLOR_BGR2HSV)
    low = np.array([0, 120, 0])
    high = np.array([14, 236, 139])

    mask = cv.inRange(copy_img, low, high)

    mask_median_blur = cv.medianBlur(mask, 3)
    mask_gausian_blur = cv.GaussianBlur(mask_median_blur, (0,0), 5)
    mask_sharpened = cv.addWeighted(mask_median_blur, 1.2, mask_gausian_blur, -0.8, 0)
    _, thresh = cv.threshold(mask_sharpened, 70, 255, cv.THRESH_BINARY)
    
    thresh = cv.bitwise_not(thresh)

    kernel = np.ones((21, 21), np.uint8)
    
    thresh = cv.erode(thresh, kernel)
    thresh = cv.dilate(thresh, kernel)

    contours, _ = cv.findContours(thresh,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
   
    for i in range(len(contours)):
        if(len(contours[i]) >3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1] :
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis = 1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    width = 1700
    height = 1700

    puzzle = np.array([[top_left,top_right,bottom_right,bottom_left]],dtype=np.float32)
    dest = np.array([[0,0],[width,0],[width,height],[0,height]],dtype=np.float32)

    M = cv.getPerspectiveTransform(puzzle,dest)
    result = cv.warpPerspective(img,M,(width,height))

    return result

def get_boards(images):
    boards = []
    for i in tqdm(range(len(images)), desc="Computing boards"):
        boards.append(get_board(images[i]))
    return boards

def get_piece_mask(img):
    copy_img = deepcopy(img)
    copy_img = cv.cvtColor(copy_img, cv.COLOR_BGR2HSV)
    low = np.array([10, 0, 180])
    high = np.array([80, 70, 220])

    mask = cv.inRange(copy_img, low, high)

    mask_median_blur = cv.medianBlur(mask, 3)
    mask_gausian_blur = cv.GaussianBlur(mask_median_blur, (0,0), 5)
    mask_sharpened = cv.addWeighted(mask_median_blur, 1.2, mask_gausian_blur, -0.8, 0)
    _, thresh = cv.threshold(mask_sharpened, 30, 255, cv.THRESH_BINARY)

    return thresh

def get_pieces_masks(images):
    piece_mask = []
    for i in tqdm(range(len(images)), desc="Computing piece masks"):
        piece_mask.append(get_piece_mask(images[i]))
    return piece_mask

def get_frame_diff(initial, next):
    diff = cv.absdiff(initial, next)

    diff_median_blur = cv.medianBlur(diff, 3)
    diff_gausian_blur = cv.GaussianBlur(diff_median_blur, (0,0), 5)
    diff_sharpened = cv.addWeighted(diff_median_blur, 1.2, diff_gausian_blur, -0.8, 0)

    _, diff_thresh = cv.threshold(diff_sharpened, 30, 255, cv.THRESH_BINARY)

    kernel1 = np.ones((7,7), np.uint8)

    diff_eroded = cv.erode(diff_thresh, kernel1)
    diff_dialated = cv.dilate(diff_eroded, kernel1)

    kernel2 = np.ones((5,5), np.uint8)

    diff_eroded2 = cv.erode(diff_dialated, kernel2)
    diff_final = cv.dilate(diff_eroded2, kernel2)

    return diff_final

def get_lines(img):
    copy_img = deepcopy(img)

    copy_img = cv.cvtColor(copy_img, cv.COLOR_HSV2BGR)

    low = np.array([30, 220, 165])
    high =np.array([150, 245, 210])

    copy_img = cv.cvtColor(copy_img, cv.COLOR_BGR2HSV)

    mask = cv.inRange(copy_img, low, high)

    image_m_blur = cv.medianBlur(mask,3)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5) 
    image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8, 0)

    _, thresh = cv.threshold(image_sharpened, 30, 255, cv.THRESH_BINARY)

    kernel = np.ones((9, 9), np.uint8)
    thresh = cv.erode(thresh, kernel)

    edges =  cv.Canny(thresh ,200,400)

    lines = cv.HoughLines(edges, 1, np.pi / 2, 130, None, 0, 0)

    lines_y = [line for line in lines if line[0][1] == 0]
    lines_x = [line for line in lines if line[0][1] != 0]

    return combine_lines(lines_x, 30), combine_lines(lines_y, 40)

def show_lines(img, lines, gray):
    copy_img = deepcopy(img)

    if gray:
        copy_img = cv.cvtColor(copy_img, cv.COLOR_GRAY2BGR)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 10000*(-b)), int(y0 + 10000*(a)))
            pt2 = (int(x0 - 10000*(-b)), int(y0 - 10000*(a)))
            cv.line(copy_img, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    show_img(copy_img)

def combine_lines(lines, thresh):

    lines_sorted = np.sort(np.array(lines), axis = 0)
    no_lines = lines_sorted.__len__()

    new_lines = []
    i, j = (0,1)

    while j != no_lines:
        if abs(lines_sorted[j][0][0] - lines_sorted[i][0][0]) < thresh:
            j += 1

            if j == no_lines:
                sum = 0
                for k in range(i,j):
                    sum += lines_sorted[k][0][0]
                new_lines.append(np.array([[sum // (j - i), lines_sorted[0][0][1]]]))

        else:
            sum = 0
            for k in range(i,j):
                sum += lines_sorted[k][0][0]
            new_lines.append(np.array([[sum // (j - i), lines_sorted[0][0][1]]]))
            i = j
            j += 1

            if j == no_lines:
                sum = 0
                for k in range(i,j):
                    sum += lines_sorted[k][0][0]
                new_lines.append(np.array([[sum // (j - i), lines_sorted[0][0][1]]]))

    
    return new_lines

def get_biggest_contour(img):
    contours, _ = cv.findContours(img,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
    i_max = 0
   
    for i in range(len(contours)):
        if(len(contours[i]) >3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1] :
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis = 1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left
                i_max = i
    return [top_left, top_right, bottom_left, bottom_right], contours[i_max]

def get_bounding_box(contour):
    x,y,w,h = cv.boundingRect(contour)
    return x,y,w,h

def show_bounding_box(x,y,w,h,img, gray = 0):
    copy_img = deepcopy(img)

    if gray:
        copy_img = cv.cvtColor(copy_img, cv.COLOR_GRAY2BGR)
    cv.rectangle(copy_img, (x,y), (x + w, y + h), color=(0,0,255), thickness= 10)
    show_img(copy_img)

def show_contour(points, img, gray = 0):
    image_copy = img.copy()
    if gray:
        image_copy = cv.cvtColor(image_copy,cv.COLOR_GRAY2BGR)
    cv.circle(image_copy,tuple(points[0]),20,(0,0,255),-1)
    cv.circle(image_copy,tuple(points[1]),20,(0,0,255),-1)
    cv.circle(image_copy,tuple(points[2]),20,(0,0,255),-1)
    cv.circle(image_copy,tuple(points[3]),20,(0,0,255),-1)
    show_img(image_copy)

def get_center_coords(x, y, w, h):
    p1 = (x, y)
    p2 = (x + w, y)
    p3 = (x, y + h)
    p4 = (x + w, y + h)

    return (p1[0] + p2[0] + p3[0] + p4[0]) // 4 , (p1[1] + p2[1] + p3[1] + p4[1]) // 4

def get_col_and_line(x, y, lines_x, lines_y):

    col, line = (0, 0)

    while y >= lines_x[line][0][0]:
        line += 1
    
    while x >= lines_y[col][0][0]:
        col += 1
    
    return line, col

def col2char(col):
    return chr(ord('A') + col - 1)

def show_point(x, y, img, gray = 0):
    copy_img = img.copy()

    if gray:
        copy_img = cv.cvtColor(copy_img, cv.COLOR_GRAY2BGR)
    
    cv.circle(copy_img, (x, y), 20, (0, 0, 255), -1)
    show_img(copy_img)

def bounding_box2perspective_transform(x, y, w, h, img):
    width = 500
    height = 500

    puzzle = np.array([[(x, y), (x + w, y), (x + w, y + h), (x, y + h)]],dtype=np.float32)
    dest = np.array([[0,0],[width,0],[width,height],[0,height]],dtype=np.float32)

    M = cv.getPerspectiveTransform(puzzle,dest)
    result = cv.warpPerspective(img,M,(width,height))

    return result

num_games = int(input())
empty_board = cv.imread("C:\\Users\\Adi\\Desktop\\Programare\\Python\\CAVA\\Labs\\Tema1\\01.jpg", cv.IMREAD_COLOR)
empty_board = get_board(empty_board)
empty_board_piece_mask = get_piece_mask(empty_board)

lines_x, lines_y = get_lines(empty_board)

path = "C:\\Users\\Adi\\Desktop\\Programare\\Python\\CAVA\\Labs\\Tema1\\antrenare\\*.jpg"
templates_path = "C:\\Users\\Adi\\Desktop\\Programare\\Python\\CAVA\\Labs\\Tema1\\templates\\*.png"
images = import_images(path, True)
templates = import_images(templates_path, True)

for temp in range(len(templates)):
    templates[temp] = cv.resize(cv.cvtColor(templates[temp], cv.COLOR_BGR2GRAY), (80,80))

index2piece = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36,40, 42, 45, 48, 49, 50, 54, 56, 60, 63, 64, 70, 72, 80, 81, 90]


dict_templates = {}
for index in range(len(templates)):
    dict_templates[index2piece[index]] = templates[index]

boards = get_boards(images)

piece_masks = get_pieces_masks(boards)

games = []
 
for i in tqdm(range(num_games)):
    games.append((boards[i*50: i*50 + 50], piece_masks[i*50: i*50 + 50]))
    games[i][0].insert(0, empty_board)
    games[i][1].insert(0, empty_board_piece_mask)

for i in tqdm(range(len(games[0][1]) - 1)):
    frame_diff = get_frame_diff(games[0][1][i], games[0][1][i+1])
    #show_img(frame_diff)
    #show_lines(frame_diff, lines_x, 1)
    #show_lines(frame_diff, lines_y, 1)
    points, contour = get_biggest_contour(frame_diff)
    #show_contour(points, frame_diff, gray = 1)
    x,y,w,h = get_bounding_box(contour)
    #show_bounding_box(x,y,w,h,frame_diff, gray=1)
    x_center, y_center = get_center_coords(x, y, w, h)
    #show_point(x_center, y_center, frame_diff, gray= 1)
    line, col = get_col_and_line(x_center, y_center, lines_x, lines_y)
    print()
    print(f"{line}{col2char(col)}")
    piece_persp_transform = bounding_box2perspective_transform(x, y, w, h, games[0][0][i+1])

    piece_persp_transform = cv.cvtColor(piece_persp_transform, cv.COLOR_BGR2GRAY)

    max_score = 0
    maxj = 0
    posible_templates = get_possible_templates(line, col)
    print(posible_templates)
    for j in posible_templates:
        res = cv.matchTemplate(piece_persp_transform, dict_templates[j], cv.TM_CCORR_NORMED) #Schimba templates[j - 1] ca ala e al 31 lea tempalte nu template lui 32
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        print(j, max_val)
        if max_val > max_score:
            max_score = max_val
            maxj = j
    print(max_score)
    print(maxj)
    show_img(dict_templates[maxj])
    piece_board[line - 1][col - 1] = maxj
   

