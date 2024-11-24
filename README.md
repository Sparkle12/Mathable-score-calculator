# Mathable score calculator
This project is a score calculator for the Scrabble like game Mathable, using various computer vision techniques to identify where and what piece was placed at each step. This is achieved
using HSV filtering, perspective transformations, contour finding, edge detection, hough trasform, motion detection, noise reduction using erosion and dilation, applying median and 
gaussian filters in order to sharpen the image and template matching. After the piece is correctly identified it is placed inside a 2D array that keeps track of the board state, and the score is
calculated

## Key takeaways
This project helped me better understand how to manipulate images and the techniques I have at my disposal in order to extract information from them. It made me understand differnt topics such as
filters, image gradients, edge detection and the difference between color representations such as BGR vs HSV
