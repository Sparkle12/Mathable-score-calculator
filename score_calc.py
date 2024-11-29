from copy import deepcopy
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from helpers import *
from tqdm import tqdm
import math
import os

dir = ".\\352_Lutu_Adrian-Catalin"

if not os.path.exists(dir):
    os.makedirs(dir)

EMPTY = -1

ANY = 0
PLUS = 1
MINUS = 2
MULTIPLY = 3
DIVIDE = 4

initial_piece_board = [
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

pieces = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 50, 54, 56, 60, 63, 64, 70, 72, 80, 81, 90]
values = [1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

initial_pieces_dict = {pieces[i] : values[i] for i in range(len(pieces))}

def append_if_valid(piece, posible):
    if piece in pieces and pieces_dict[piece] > 0:
        posible.append(piece)

def get_possible_templates(line, col, piece_board):
    i = line - 1
    j = col - 1
    posible = []
    if sign_board[i][j] == ANY:
        if i >= 2:
            if piece_board[i - 1][j] != EMPTY and piece_board[i - 2][j] != EMPTY:
                append_if_valid(piece_board[i - 1][j] + piece_board[i - 2][j], posible)
                append_if_valid(piece_board[i - 1][j] * piece_board[i - 2][j], posible)
                append_if_valid(abs(piece_board[i - 1][j] - piece_board[i - 2][j]), posible)
                if piece_board[i - 2][j] != 0 and piece_board[i - 1][j] / piece_board[i - 2][j] == piece_board[i - 1][j] // piece_board[i - 2][j]:
                    append_if_valid(piece_board[i-1][j] // piece_board[i-2][j], posible)
                elif piece_board[i - 1][j] != 0 and piece_board[i - 2][j] / piece_board[i - 1][j] == piece_board[i - 2][j] // piece_board[i - 1][j]:
                    append_if_valid(piece_board[i - 2][j] // piece_board[i - 1][j], posible)
        if i <= 11:
            if piece_board[i + 1][j] != EMPTY and piece_board[i + 2][j] != EMPTY:
                append_if_valid(piece_board[i + 1][j] + piece_board[i + 2][j], posible)
                append_if_valid(piece_board[i + 1][j] * piece_board[i + 2][j], posible)
                append_if_valid(abs(piece_board[i + 1][j] - piece_board[i + 2][j]), posible)
                if piece_board[i + 2][j] != 0 and piece_board[i + 1][j] / piece_board[i + 2][j] == piece_board[i + 1][j] // piece_board[i + 2][j]:
                    append_if_valid(piece_board[i + 1][j] // piece_board[i + 2][j], posible)
                elif piece_board[i + 1][j] != 0 and piece_board[i + 2][j] / piece_board[i + 1][j] == piece_board[i + 2][j] // piece_board[i + 1][j]:
                    append_if_valid(piece_board[i + 2][j] // piece_board[i + 1][j], posible)
        if j >= 2:
            if piece_board[i][j - 1] != EMPTY and piece_board[i][j - 2] != EMPTY:
                append_if_valid(piece_board[i][j - 1] + piece_board[i][j - 2], posible)
                append_if_valid(piece_board[i][j - 1] * piece_board[i][j - 2], posible)
                append_if_valid(abs(piece_board[i][j - 1] - piece_board[i][j - 2]), posible)
                if piece_board[i][j - 2] != 0 and piece_board[i][j - 1] / piece_board[i][j - 2] == piece_board[i][j - 1] // piece_board[i][j - 2]:
                    append_if_valid(piece_board[i][j - 1] // piece_board[i][j - 2], posible)
                elif piece_board[i][j - 1] != 0 and piece_board[i][j - 2] / piece_board[i][j - 1] == piece_board[i][j - 2] // piece_board[i][j - 1]:
                    append_if_valid(piece_board[i][j - 2] // piece_board[i][j - 1], posible)
        if j <= 11:
            if piece_board[i][j + 1] != EMPTY and piece_board[i][j + 2] != EMPTY:
                append_if_valid(piece_board[i][j + 1] + piece_board[i][j + 2], posible)
                append_if_valid(piece_board[i][j + 1] * piece_board[i][j + 2], posible)
                append_if_valid(abs(piece_board[i][j + 1] - piece_board[i][j + 2]), posible)
                if piece_board[i][j + 2] != 0 and piece_board[i][j + 1] / piece_board[i][j + 2] == piece_board[i][j + 1] // piece_board[i][j + 2]:
                    append_if_valid(piece_board[i][j + 1] // piece_board[i][j + 2], posible)
                elif piece_board[i][j + 1] != 0 and piece_board[i][j + 2] / piece_board[i][j + 1] == piece_board[i][j + 2] // piece_board[i][j + 1]:
                    append_if_valid(piece_board[i][j + 2] // piece_board[i][j + 1], posible)
    elif sign_board[i][j] == MULTIPLY:
        if i >= 2:
            if piece_board[i - 1][j] != EMPTY and piece_board[i - 2][j] != EMPTY:
                append_if_valid(piece_board[i - 1][j] * piece_board[i - 2][j], posible)
        if i <= 11:
            if piece_board[i + 1][j] != EMPTY and piece_board[i + 2][j] != EMPTY:
                append_if_valid(piece_board[i + 1][j] * piece_board[i + 2][j], posible)
        if j >= 2:
            if piece_board[i][j - 1] != EMPTY and piece_board[i][j - 2] != EMPTY:
                append_if_valid(piece_board[i][j - 1] * piece_board[i][j - 2], posible)
        if j <= 11:
            if piece_board[i][j + 1] != EMPTY and piece_board[i][j + 2] != EMPTY:
                append_if_valid(piece_board[i][j + 1] * piece_board[i][j + 2], posible)
    elif sign_board[i][j] == PLUS:
        if i >= 2:
            if piece_board[i - 1][j] != EMPTY and piece_board[i - 2][j] != EMPTY:
                append_if_valid(piece_board[i - 1][j] + piece_board[i - 2][j], posible)
        if i <= 11:
            if piece_board[i + 1][j] != EMPTY and piece_board[i + 2][j] != EMPTY:
                append_if_valid(piece_board[i + 1][j] + piece_board[i + 2][j], posible)
        if j >= 2:
            if piece_board[i][j - 1] != EMPTY and piece_board[i][j - 2] != EMPTY:
                append_if_valid(piece_board[i][j - 1] + piece_board[i][j - 2], posible)
        if j <= 11:
            if piece_board[i][j + 1] != EMPTY and piece_board[i][j + 2] != EMPTY:
                append_if_valid(piece_board[i][j + 1] + piece_board[i][j + 2], posible)
    elif sign_board[i][j] == MINUS:
        if i >= 2:
            if piece_board[i - 1][j] != EMPTY and piece_board[i - 2][j] != EMPTY:
                append_if_valid(abs(piece_board[i - 1][j] - piece_board[i - 2][j]), posible)
        if i <= 11:
            if piece_board[i + 1][j] != EMPTY and piece_board[i + 2][j] != EMPTY:
                append_if_valid(abs(piece_board[i + 1][j] - piece_board[i + 2][j]), posible)
        if j >= 2:
            if piece_board[i][j - 1] != EMPTY and piece_board[i][j - 2] != EMPTY:
                append_if_valid(abs(piece_board[i][j - 1] - piece_board[i][j - 2]), posible)
        if j <= 11:
            if piece_board[i][j + 1] != EMPTY and piece_board[i][j + 2] != EMPTY:
                append_if_valid(abs(piece_board[i][j + 1] - piece_board[i][j + 2]), posible)
    elif sign_board[i][j] == DIVIDE:
        if i >= 2:
            if piece_board[i - 1][j] != EMPTY and piece_board[i - 2][j] != EMPTY:
                if piece_board[i - 2][j] != 0 and piece_board[i - 1][j] / piece_board[i - 2][j] == piece_board[i - 1][j] // piece_board[i - 2][j]:
                    append_if_valid(piece_board[i-1][j] // piece_board[i-2][j], posible)
                elif piece_board[i - 1][j] != 0 and piece_board[i - 2][j] / piece_board[i - 1][j] == piece_board[i - 2][j] // piece_board[i - 1][j]:
                    append_if_valid(piece_board[i - 2][j] // piece_board[i - 1][j], posible)
        if i <= 11:
            if piece_board[i + 1][j] != EMPTY and piece_board[i + 2][j] != EMPTY:
                if piece_board[i + 2][j] != 0 and piece_board[i + 1][j] / piece_board[i + 2][j] == piece_board[i + 1][j] // piece_board[i + 2][j]:
                    append_if_valid(piece_board[i + 1][j] // piece_board[i + 2][j], posible)
                elif piece_board[i + 1][j] != 0 and piece_board[i + 2][j] / piece_board[i + 1][j] == piece_board[i + 2][j] // piece_board[i + 1][j]:
                    append_if_valid(piece_board[i + 2][j] // piece_board[i + 1][j], posible)
        if j >= 2:
            if piece_board[i][j - 1] != EMPTY and piece_board[i][j - 2] != EMPTY:
                if piece_board[i][j - 2] != 0 and piece_board[i][j - 1] / piece_board[i][j - 2] == piece_board[i][j - 1] // piece_board[i][j - 2]:
                    append_if_valid(piece_board[i][j - 1] // piece_board[i][j - 2], posible)
                elif piece_board[i][j - 1] != 0 and piece_board[i][j - 2] / piece_board[i][j - 1] == piece_board[i][j - 2] // piece_board[i][j - 1]:
                    append_if_valid(piece_board[i][j - 2] // piece_board[i][j - 1], posible)
        if j <= 11:
            if piece_board[i][j + 1] != EMPTY and piece_board[i][j + 2] != EMPTY:
                if piece_board[i][j + 2] != 0 and piece_board[i][j + 1] / piece_board[i][j + 2] == piece_board[i][j + 1] // piece_board[i][j + 2]:
                    append_if_valid(piece_board[i][j + 1] // piece_board[i][j + 2], posible)
                elif piece_board[i][j + 1] != 0 and  piece_board[i][j + 2] / piece_board[i][j + 1] == piece_board[i][j + 2] // piece_board[i][j + 1]:
                    append_if_valid(piece_board[i][j + 2] // piece_board[i][j + 1], posible)
                      
    return set(posible)


def calculate_score(line, col, piece_board, piece):
    i = line - 1
    j = col - 1
    score = 0
    if sign_board[i][j] == ANY:
        if i >= 2:
            if piece_board[i - 1][j] != EMPTY and piece_board[i - 2][j] != EMPTY:
                if piece_board[i - 1][j] + piece_board[i - 2][j] == piece:
                    score += piece * multiplier_board[i][j]
                elif piece_board[i - 1][j] * piece_board[i - 2][j] == piece:
                    score += piece * multiplier_board[i][j]
                elif abs(piece_board[i - 1][j] - piece_board[i - 2][j]) == piece:
                    score += piece * multiplier_board[i][j]
                else:
                    if piece_board[i - 2][j] != 0 and piece_board[i - 1][j] / piece_board[i - 2][j] == piece_board[i - 1][j] // piece_board[i - 2][j]:
                        if piece_board[i-1][j] // piece_board[i-2][j] == piece:
                            score += piece * multiplier_board[i][j]
                    elif piece_board[i - 1][j] != 0 and piece_board[i - 2][j] / piece_board[i - 1][j] == piece_board[i - 2][j] // piece_board[i - 1][j]:
                        if piece_board[i - 2][j] // piece_board[i - 1][j] == piece:
                            score += piece * multiplier_board[i][j]
        if i <= 11:
            if piece_board[i + 1][j] != EMPTY and piece_board[i + 2][j] != EMPTY:
                if piece_board[i + 1][j] + piece_board[i + 2][j] == piece:
                    score += piece * multiplier_board[i][j]
                elif piece_board[i + 1][j] * piece_board[i + 2][j] == piece:
                    score += piece * multiplier_board[i][j]
                elif abs(piece_board[i + 1][j] - piece_board[i + 2][j]) == piece:
                    score += piece * multiplier_board[i][j]
                else:
                    if piece_board[i + 2][j] != 0 and piece_board[i + 1][j] / piece_board[i + 2][j] == piece_board[i + 1][j] // piece_board[i + 2][j]:
                        if piece_board[i+1][j] // piece_board[i+2][j] == piece:
                            score += piece * multiplier_board[i][j]
                    elif piece_board[i + 1][j] != 0 and piece_board[i + 2][j] / piece_board[i + 1][j] == piece_board[i + 2][j] // piece_board[i + 1][j]:
                        if piece_board[i + 2][j] // piece_board[i + 1][j] == piece:
                            score += piece * multiplier_board[i][j]
        if j >= 2:
            if piece_board[i][j - 1] != EMPTY and piece_board[i][j - 2] != EMPTY:
                if piece_board[i][j - 1] + piece_board[i][j - 2] == piece:
                    score += piece * multiplier_board[i][j]
                elif piece_board[i][j - 1] * piece_board[i][j - 2] == piece:
                    score += piece * multiplier_board[i][j]
                elif abs(piece_board[i][j - 1] - piece_board[i][j - 2]) == piece:
                    score += piece * multiplier_board[i][j]
                else:
                    if piece_board[i][j - 2] != 0 and piece_board[i][j - 1] / piece_board[i][j - 2] == piece_board[i][j - 1] // piece_board[i][j - 2]:
                        if piece_board[i][j - 1] // piece_board[i][j - 2] == piece:
                            score += piece * multiplier_board[i][j]
                    elif piece_board[i][j - 1] != 0 and piece_board[i][j - 2] / piece_board[i][j - 1] == piece_board[i][j - 2] // piece_board[i][j - 1]:
                        if piece_board[i][j - 2] // piece_board[i][j - 1] == piece:
                            score += piece * multiplier_board[i][j]
        if j <= 11:
            if piece_board[i][j + 1] != EMPTY and piece_board[i][j + 2] != EMPTY:
                if piece_board[i][j + 1] + piece_board[i][j + 2] == piece:
                    score += piece * multiplier_board[i][j]
                elif piece_board[i][j + 1] * piece_board[i][j + 2] == piece:
                    score += piece * multiplier_board[i][j]
                elif abs(piece_board[i][j + 1] - piece_board[i][j + 2]) == piece:
                    score += piece * multiplier_board[i][j]
                else:
                    if piece_board[i][j + 2] != 0 and piece_board[i][j + 1] / piece_board[i][j + 2] == piece_board[i][j + 1] // piece_board[i][j + 2]:
                        if piece_board[i][j + 1] // piece_board[i][j + 2] == piece:
                            score += piece * multiplier_board[i][j]
                    elif piece_board[i][j + 1] != 0 and piece_board[i][j + 2] / piece_board[i][j + 1] == piece_board[i][j + 2] // piece_board[i][j + 1]:
                        if piece_board[i][j + 2] // piece_board[i][j + 1] == piece:
                            score += piece * multiplier_board[i][j]
    elif sign_board[i][j] == MULTIPLY:
        if i >= 2:
            if piece_board[i - 1][j] != EMPTY and piece_board[i - 2][j] != EMPTY:
                if piece_board[i - 1][j] * piece_board[i - 2][j] == piece:
                    score += piece * multiplier_board[i][j]
        if i <= 11:
            if piece_board[i + 1][j] != EMPTY and piece_board[i + 2][j] != EMPTY:
                if piece_board[i + 1][j] * piece_board[i + 2][j] == piece:
                    score += piece * multiplier_board[i][j]
        if j >= 2:
            if piece_board[i][j - 1] != EMPTY and piece_board[i][j - 2] != EMPTY:
                if piece_board[i][j - 1] * piece_board[i][j - 2] == piece:
                    score += piece * multiplier_board[i][j]
        if j <= 11:
            if piece_board[i][j + 1] != EMPTY and piece_board[i][j + 2] != EMPTY:
                if piece_board[i][j + 1] * piece_board[i][j + 2] == piece:
                    score += piece * multiplier_board[i][j]
    elif sign_board[i][j] == PLUS:
        if i >= 2:
            if piece_board[i - 1][j] != EMPTY and piece_board[i - 2][j] != EMPTY:
                if piece_board[i - 1][j] + piece_board[i - 2][j] == piece:
                    score += piece * multiplier_board[i][j]
        if i <= 11:
            if piece_board[i + 1][j] != EMPTY and piece_board[i + 2][j] != EMPTY:
                if piece_board[i + 1][j] + piece_board[i + 2][j] == piece:
                    score += piece * multiplier_board[i][j]
        if j >= 2:
            if piece_board[i][j - 1] != EMPTY and piece_board[i][j - 2] != EMPTY:
                if piece_board[i][j - 1] + piece_board[i][j - 2] == piece:
                    score += piece * multiplier_board[i][j]
        if j <= 11:
            if piece_board[i][j + 1] != EMPTY and piece_board[i][j + 2] != EMPTY:
                if piece_board[i][j + 1] + piece_board[i][j + 2] == piece:
                    score += piece * multiplier_board[i][j]
    elif sign_board[i][j] == MINUS:
        if i >= 2:
            if piece_board[i - 1][j] != EMPTY and piece_board[i - 2][j] != EMPTY:
                if abs(piece_board[i - 1][j] - piece_board[i - 2][j]) == piece:
                    score += piece * multiplier_board[i][j]
        if i <= 11:
            if piece_board[i + 1][j] != EMPTY and piece_board[i + 2][j] != EMPTY:
                if abs(piece_board[i + 1][j] - piece_board[i + 2][j]) == piece:
                    score += piece * multiplier_board[i][j]
        if j >= 2:
            if piece_board[i][j - 1] != EMPTY and piece_board[i][j - 2] != EMPTY:
                if abs(piece_board[i][j - 1] - piece_board[i][j - 2]) == piece:
                    score += piece * multiplier_board[i][j]
        if j <= 11:
            if piece_board[i][j + 1] != EMPTY and piece_board[i][j + 2] != EMPTY:
                if abs(piece_board[i][j + 1] - piece_board[i][j + 2]) == piece:
                    score += piece * multiplier_board[i][j]
    elif sign_board[i][j] == DIVIDE:
        if i >= 2:
            if piece_board[i - 1][j] != EMPTY and piece_board[i - 2][j] != EMPTY:
                if piece_board[i - 2][j] != 0 and piece_board[i - 1][j] / piece_board[i - 2][j] == piece_board[i - 1][j] // piece_board[i - 2][j]:
                    if piece_board[i-1][j] // piece_board[i-2][j] == piece:
                        score += piece * multiplier_board[i][j]
                elif piece_board[i - 1][j] != 0 and piece_board[i - 2][j] / piece_board[i - 1][j] == piece_board[i - 2][j] // piece_board[i - 1][j]:
                    if piece_board[i - 2][j] // piece_board[i - 1][j] == piece:
                        score += piece * multiplier_board[i][j]
        if i <= 11:
            if piece_board[i + 1][j] != EMPTY and piece_board[i + 2][j] != EMPTY:
                if piece_board[i + 2][j] != 0 and piece_board[i + 1][j] / piece_board[i + 2][j] == piece_board[i + 1][j] // piece_board[i + 2][j]:
                    if piece_board[i + 1][j] // piece_board[i + 2][j] == piece:
                        score += piece * multiplier_board[i][j]
                elif piece_board[i + 1][j] != 0 and piece_board[i + 2][j] / piece_board[i + 1][j] == piece_board[i + 2][j] // piece_board[i + 1][j]:
                    if piece_board[i + 2][j] // piece_board[i + 1][j] == piece:
                        score += piece * multiplier_board[i][j]
        if j >= 2:
            if piece_board[i][j - 1] != EMPTY and piece_board[i][j - 2] != EMPTY:
                if piece_board[i][j - 2] != 0 and piece_board[i][j - 1] / piece_board[i][j - 2] == piece_board[i][j - 1] // piece_board[i][j - 2]:
                    if piece_board[i][j - 1] // piece_board[i][j - 2] == piece:
                        score += piece * multiplier_board[i][j]
                elif piece_board[i][j - 1] != 0 and piece_board[i][j - 2] / piece_board[i][j - 1] == piece_board[i][j - 2] // piece_board[i][j - 1]:
                    if piece_board[i][j - 2] // piece_board[i][j - 1] == piece:
                        score += piece * multiplier_board[i][j]
        if j <= 11:
            if piece_board[i][j + 1] != EMPTY and piece_board[i][j + 2] != EMPTY:
                if piece_board[i][j + 2] != 0 and piece_board[i][j + 1] / piece_board[i][j + 2] == piece_board[i][j + 1] // piece_board[i][j + 2]:
                    if piece_board[i][j + 1] // piece_board[i][j + 2] == piece:
                        score += piece * multiplier_board[i][j]
                elif piece_board[i][j + 1] != 0 and  piece_board[i][j + 2] / piece_board[i][j + 1] == piece_board[i][j + 2] // piece_board[i][j + 1]:
                    if piece_board[i][j + 2] // piece_board[i][j + 1] == piece:
                        score += piece * multiplier_board[i][j]
                      
    return score

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

    kernel_erode = np.ones((25, 25), np.uint8)
    
    thresh = cv.erode(thresh, kernel_erode)

    #kernel_dilate = np.ones((13, 13), np.uint8)
    #thresh = cv.dilate(thresh, kernel_dilate)

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

    return diff_dialated

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
    return [top_left, top_right, bottom_right, bottom_left], contours[i_max]

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
    width = 140
    height = 140

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

path = "C:\\Users\\Adi\\Desktop\\CAVA-2024-Tema1\\evaluare\\fake_test\\*.jpg"
templates_path = "C:\\Users\\Adi\\Desktop\\Programare\\Python\\CAVA\\Labs\\Tema1\\templates_cropped\\*.png"
#txt_files = glob.glob("C:\\Users\\Adi\\Desktop\\Programare\\Python\\CAVA\\Labs\\Tema1\\antrenare\\*.txt")

turns_files = glob.glob("C:\\Users\\Adi\\Desktop\\CAVA-2024-Tema1\\evaluare\\fake_test\\*_turns.txt")

game_turns = []
game_players = []

for file in turns_files:
    f = open(file, 'r')
    string = f.read()
    rows = string.split('\n')
    turns = [int(x.split()[-1]) for x in rows]
    players = [int(x.split()[0][-1]) - 1 for x in rows]
    turns.append(51)
    game_turns.append(turns)
    game_players.append(players[0])

print(game_players)

images = import_images(path, True)
templates = import_images(templates_path, True)

dict_templates = {}
for index in range(len(templates)):
    dict_templates[pieces[index]] = templates[index]

boards = get_boards(images)

piece_masks = get_pieces_masks(boards)

games = []
 
for i in tqdm(range(num_games)):
    games.append((boards[i*50: i*50 + 50], piece_masks[i*50: i*50 + 50]))
    games[i][0].insert(0, empty_board)
    games[i][1].insert(0, empty_board_piece_mask)

for game in tqdm(range(num_games)):
    piece_board = deepcopy(initial_piece_board)
    pieces_dict = deepcopy(initial_pieces_dict)
    switch_turn_index = 1
    player = game_players[game]
    turn_score = 0
    scores_file = open(dir + f"\\{game + 1}_scores.txt", "a")
    for i in range(len(games[game][1]) - 1):
        if i + 1 == game_turns[game][switch_turn_index]:
            scores_file.write(f"Player{player + 1} {game_turns[game][switch_turn_index - 1]} {turn_score}\n")
            player = (player + 1) % 2
            switch_turn_index += 1
            turn_score = 0
        
        frame_diff = get_frame_diff(games[game][1][i], games[game][1][i+1])
        show_img(frame_diff)
        show_lines(frame_diff, lines_x, 1)
        show_lines(frame_diff, lines_y, 1)
        points, contour = get_biggest_contour(frame_diff)
        show_contour(points, frame_diff, gray = 1)
        x,y,w,h = get_bounding_box(contour)
        show_bounding_box(x,y,w,h,frame_diff, gray=1)
        x_center, y_center = get_center_coords(x, y, w, h)
        show_point(x_center, y_center, frame_diff, gray= 1)
        line, col = get_col_and_line(x_center, y_center, lines_x, lines_y)
        piece_persp_transform = bounding_box2perspective_transform(x, y, w, h, games[game][0][i+1])

        show_img(piece_persp_transform)
        max_score = 0
        maxj = 0
        top_left = None
        bottom_right = None
        posible_templates = get_possible_templates(line, col, piece_board)
        vals = []
        for j in posible_templates:
            h, w, _ = dict_templates[j].shape
            res = cv.matchTemplate(piece_persp_transform, dict_templates[j], cv.TM_CCORR_NORMED) 
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            vals.append((j, max_val))

            if max_val > max_score:
                max_score = max_val
                maxj = j
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)

        piece_board[line - 1][col - 1] = maxj
        pieces_dict[maxj] -= 1
        turn_score += calculate_score(line, col, piece_board, maxj)
        path = dir + f"\\{game + 1}_{i+1:02d}.txt"
        piece_file = open(path, 'w')
        piece_file.write(f"{line}{col2char(col)} {maxj}")
        piece_file.close()
    scores_file.write(f"Player{player + 1} {game_turns[game][switch_turn_index - 1]} {turn_score}")
    scores_file.close()
