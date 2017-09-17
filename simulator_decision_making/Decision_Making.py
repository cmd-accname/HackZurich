import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def auto_canny(image, sigma=0.33):
    """Applies Canny transform and calculates the threshold"""
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image
	
def decide_turn(img, lines, min_slope, max_slope, color=[255, 0, 0], thickness=11):
    if lines is None:
        return 'no decision'

    x_middle = img.shape[1]/2
    right_b = None
    right_a = None

    y_min = 320
    y_max = img.shape[0]

    l_slope = []
    l_coeff = []

    r_slope = []
    r_coeff = []

    l_lane = np.empty([0,2],int)
    r_lane = np.empty([0,2],int)
    
    print(lines)

    for line in lines:
        for x1,y1,x2,y2 in line:
            # calculate the slope
            m = ((y2-y1)/(x2-x1))
            # calculate the free coefficient   
            b = y1 - m*x1
            # filter the slope for left lane
            if m > -max_slope and m < -min_slope:
                l_slope.append(m)
                l_coeff.append(b)
                l_lane= np.append(l_lane,[[x1,y1],[x2,y2]], axis = 0)
            # filter the slope for right lane    
            elif m > min_slope and m < max_slope:
                r_slope.append(m)
                r_coeff.append(b)
                r_lane= np.append(r_lane,[[x1,y1],[x2,y2]], axis = 0)

    if len(l_slope) == 0:
        if len(r_slope) == 0:
            return 'no decision'
        else:
            return 'left'
    elif len(r_slope) == 0:
        return 'right'

    # if we reach here, both slope-sets exist!    

    l_slopemean = np.mean(l_slope) 
    l_coeffmean = np.mean(l_coeff)
    l_mean = np.mean(l_lane, axis = 0)
    left_b = l_mean[1] - l_slopemean*l_mean[0]
    left_x1 = int((y_min - left_b)/l_slopemean)
    left_x2 = int((y_max - left_b)/l_slopemean)
    cv2.line(img, (left_x1, y_min), (left_x2, y_max), color, thickness)

    r_slopemean = np.mean(r_slope)
    r_coeffmean = np.mean(r_coeff)
    r_mean = np.mean(r_lane, axis = 0)
    right_b = r_mean[1] - r_slopemean*r_mean[0]
    right_x1 = int((y_min - right_b)/r_slopemean)
    right_x2 = int((y_max - right_b)/r_slopemean)
    cv2.line(img, (right_x1, y_min), (right_x2, y_max), color, thickness)

    #calculate x intercept of two lines
    x_intercept = (right_b - left_b) / (l_slopemean - r_slopemean)  
    if x_intercept > x_middle + 10:
        return 'right'

    elif x_intercept < x_middle - 10:
        return 'left'
        
    else:
        return 'no decision'

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    `initial_img` should be the image before any processing.
    The result image is computed as follows:
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_frame(image, min_slope, max_slope):
    gray = grayscale(image)
    # Apply gaussian smoothing and canny edge detection
    blur_gray = gaussian_blur(gray, 5)
    edges = canny(blur_gray, 50, 150)

    # Define the Hough transform parameters
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 100     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 100 #minimum number of pixels making up a line
    max_line_gap = 4    # maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*edges.shape, 3), dtype=np.uint8)
    color=[255, 0, 0]
    thickness=11

    decision = decide_turn(line_img, lines, min_slope, max_slope)

    if lines is None:
        return decision, image
    else:
        for line in lines:

            for x1,y1,x2,y2 in line:

                cv2.line(image, (x1,y1), (x2,y2), color, thickness)

    return decision, image       


# Read the image from directory and convert to gray scale

# for i in range(300,400):

    # image = cv2.imread("../data/data"+str(i)+".png")

    # decision, image = process_frame(image,0.01,30)

    # plt.figure()
    # plt.imshow(image)
    # plt.show()
    #plt.imshow(image)
    # print(decision)