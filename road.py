import cv2
import numpy as np


def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def region_of_interest(img):
    height = img.shape[0]

    triangle = np.array([[(-20, 90), (320, height), (130, 40)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, triangle, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def display_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 5)
    return line_img


def make_coordinates(img, line_parameters):
    slope, intercept = line_parameters
    y1 = img.shape[0]
    y2 = int(y1 * 0.85)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(img, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope > 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    left_line = make_coordinates(img, left_fit_avg)
    right_line = make_coordinates(img, right_fit_avg)
    return np.array([left_line, right_line])


image = cv2.imread("image.jpg")
lane_image = np.copy(image)
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)
lines = cv2.HoughLinesP(
    cropped_image, 2, np.pi / 180, 1, np.array([]), minLineLength=20, maxLineGap=5
)
average_lines = average_slope_intercept(lane_image, lines)
line_image = display_lines(lane_image, average_lines)
combo_image = cv2.addWeighted(lane_image, 1, line_image, 1, 1)

cv2.imshow("img", combo_image)
cv2.waitKey(0)
