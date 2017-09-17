import cv2
import numpy as np



# image is expected be in RGB color space
def select_rgb_white(image): 
    # white color mask
    lower = np.uint8([200, 200, 200])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)

    # combine the mask
    masked = cv2.bitwise_and(image, image, mask = white_mask)
    return masked

def select_hls(image):
	# white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hls, lower, upper)

    return cv2.bitwise_and(hls, hls, mask = mask)

def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
    return cv2.bitwise_and(image, mask)

    
def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left  = [0, rows]
    top_left     = [0, rows*0.4]
    bottom_right = [cols, rows]
    top_right    = [cols, rows*0.4]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)

def average_slope_intercept(lines):
    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2==x1:
                continue # ignore a vertical line
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0: # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    
    # add more weight to longer lines    
    left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
    
    return left_lane, right_lane # (slope, intercept), (slope, intercept)

def make_line_points(y1, y2, line):
	"""
	Convert a line represented in slope and intercept into pixel points
	"""
	if line is None:
		return None

	slope, intercept = line

	# make sure everything is integer as cv2.line requires it
	x1 = (y1 - intercept)/slope
	if (np.isinf(x1)):
		return None
	x2 = (y2 - intercept)/slope
	if (np.isinf(x2)):
		return None
	x1 = int(x1)
	x2 = int(x2)
	y1 = int(y1)
	y2 = int(y2)

	return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
	if (lines is None):
		return None, None
	left_lane, right_lane = average_slope_intercept(lines)


	y1 = image.shape[0] # bottom of the image
	y2 = y1*0.6         # slightly lower than the middle

	if (left_lane is not None):
		left_line  = make_line_points(y1, y2, left_lane)
	else:
		left_line = None
	if (right_lane is not None):
		right_line = make_line_points(y1, y2, right_lane)
	else:
		right_line = None

	return left_line, right_line

    
def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=20):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(image, *line,  color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return image


def decide_turn(img, lines, min_slope, max_slope, angle=1, color=[255, 0, 0], thickness=11):

    """

    NOTE: this is the function you might want to use as a starting point once you want to 

    average/extrapolate the line segments you detect to map out the full

    extent of the lane (going from the result shown in raw-lines-example.mp4

    to that shown in P1_example.mp4).  

    

    Think about things like separating line segments by their 

    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left

    line vs. the right line.  Then, you can average the position of each of 

    the lines and extrapolate to the top and bottom of the lane.

    

    This function draws `lines` with `color` and `thickness`.    

    Lines are drawn on the image inplace (mutates the image).

    If you want to make the lines semi-transparent, think about combining

    this function with the weighted_img() function below

    """

    if lines[0] == None: #left is none
        if lines[1] == None: # right is none
            return 0
        else:
            right = sum(lines[1], ())
            r_slope = ((right[3]-right[1])/(right[2]-right[0]))
            r_coeff = right[1]-r_slope*right[0]
            # print(r_slope)
            return -angle
    elif lines[1] == None: #right is none
        left = sum(lines[0], ())
        l_slope = ((left[3]-left[1])/(left[2]-left[0]))
        l_coeff = left[1]-l_slope*left[0]
        # print(l_slope)
        return angle

    # if we reach here, both slope-sets exist!  


    x_middle = img.shape[1]/2
    
    y_min = 320
    y_max = img.shape[0]
    
    l_slope = []
    l_coeff = []

    r_slope = []
    r_coeff = []

    #print(lines)

    left = sum(lines[0], ())
    l_slope = ((left[3]-left[1])/(left[2]-left[0]))
    l_coeff = left[1]-l_slope*left[0]

    right = sum(lines[1], ())
    r_slope = ((right[3]-right[1])/(right[2]-right[0]))
    r_coeff = right[1]-r_slope*right[0]

    # print(l_slope)
    #print(l_coeff)
    # print(r_slope)
    #print(r_coeff)


    #for x1,y1,x2,y2 in sum(lines[0], ()):
        # calculate the slope
        #l_slope = ((y2-y1)/(x2-x1))
        # calculate the free coefficient 
        #l_coeff = y1 - m*x1    

   # for x1,y1,x2,y2 in sum(lines[1], ()):
        # calculate the slope
    #    r_slope = ((y2-y1)/(x2-x1))
        # calculate the free coefficient 
     #   r_coeff = y1 - m*x1    
     
        
    #calculate x intercept of two lines
    x_intercept = (r_coeff - l_coeff) / (l_slope - r_slope)  

    angle = x_intercept - x_middle

    return angle / img.shape[0] # * threshold



             
    
# lane_images = []

# # img = cv2.imread('data/color2.png')
# cap = cv2.VideoCapture('data/1505593738.06-output.avi')
# while(True):
# 	# Capture frame-by-frame
# 	ret, frame = cap.read()

# 	if (not ret):
# 		break

# 	# Display the resulting frame
# 	img = frame
# 	# print(img.shape)
# 	# print(np.max(img))
# 	# print(np.min(img))

# 	# masked = select_rgb_white(img)
# 	# cv2.imshow('image', masked)
# 	# cv2.waitKey(0)

# 	# hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# 	# cv2.imshow('image', hsv)
# 	# cv2.waitKey(0)

# 	# hls = select_hls(img)
# 	# cv2.imshow('image', hls)
# 	# cv2.waitKey(0)

# 	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# 	#kernel_size must be postivie and odd
# 	kernel_size = 15
# 	blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
# 	#Recommended a upper:lower ratio between 2:1 and 3:1
# 	low_threshold=50
# 	high_threshold=150
# 	canny = cv2.Canny(blurred, low_threshold, high_threshold)
# 	# cv2.imshow('image', canny)
# 	# cv2.waitKey(0)

# 	# region = select_region(img)
# 	# cv2.imshow('image', region)
# 	# cv2.waitKey(0)

# 	region = select_region(canny)
# 	# cv2.imshow('image', region)
# 	# cv2.waitKey(0)

# 	hough = cv2.HoughLinesP(region, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)
# 	lanes = lane_lines(img, hough)

# 	decision = decide_turn(img, lanes, 0.01, 30)
# 	print(decision)

# 	# print(lanes)
# 	img = draw_lane_lines(img, lanes)
# 	cv2.imshow('image', img)
# 	# cv2.waitKey(0)

# 	# cv2.destroyAllWindows()
# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()