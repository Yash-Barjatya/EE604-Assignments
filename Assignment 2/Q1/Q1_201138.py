import cv2
import numpy as np


def compute_lbp_pixel(center, pixels):
    """Computes the Local Binary Pattern (LBP) for a single pixel.

    Args:
        center: The central pixel value.
        pixels: A list of the 8 neighboring pixel values.

    Returns:
        An integer representing the LBP value.
    """
    s = np.argmax(pixels)
    n = len(pixels)
    binary_string = ""
    if pixels[s] < center:
        return 0
    else:
        binary_string += '1'

    i = (s + 1) % n
    while i != s:
        if pixels[i] < center:
            binary_string += "0"
        else:
            binary_string += '1'
        i = (i + 1) % n

    reversed_binary_string = binary_string[::-1]
    return int(reversed_binary_string, 2)


def compute_lbp_mode(img, thresholded_img, r=1):
    """Computes the most common LBP value in a local region.
    Args:
        img: The original image.
        thresholded_img: A thresholded image of the lava region.
        r: The radius of the local region.

    Returns:
        An integer representing the most common LBP value in the local region.
    """
    lbp_values = []

    rows, cols = img.shape
    for i in range(r, rows - r):
        for j in range(r, cols - r):
            center = img[i, j]
            pixel_values = [
                img[i - r, j],
                img[i - r, j + r],
                img[i, j + r],
                img[i + r, j + r],
                img[i + r, j],
                img[i + r, j - r],
                img[i, j - r],
                img[i - r, j - r]
            ]
            if thresholded_img[i][j] == 255:
                lbp_values.append(compute_lbp_pixel(
                    center, np.array(pixel_values)))

    lbp_array = np.array(lbp_values)
    frequency = {}
    for value in lbp_array:
        frequency.setdefault(value, 0)
        frequency[value] += 1

    highest_frequency = max(frequency.values())
    mode_value = 0

    for i, count in frequency.items():
        if count == highest_frequency:
            mode_value = i

    return mode_value


def get_lava_region(input_image):
    """Extracts the lava region from an input image.

    Args:
        input_image: The input image.

    Returns:
        A binary image of the lava region.
    """
    gray_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

    blurred_image = cv2.GaussianBlur(hsv_image, (13, 13), 0)

    # Define the lower and upper HSV bounds for lava.
    lower_lava = np.array([0, 120, 120])
    upper_lava = np.array([170, 255, 255])

    # Threshold the image to extract the lava region.
    lava_mask_threshold = cv2.inRange(blurred_image, lower_lava, upper_lava)

    # Find the contours in the lava mask.
    contours, _ = cv2.findContours(
        lava_mask_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fill in the largest contour.
    filled_mask = np.zeros_like(lava_mask_threshold)

    img_area = lava_mask_threshold.shape[0] * lava_mask_threshold.shape[1]
    for contour in contours:
        if cv2.contourArea(contour) >= 0.09 * img_area:
            temp_filled_mask = np.zeros_like(lava_mask_threshold)
            cv2.drawContours(
                temp_filled_mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)
            mode = compute_lbp_mode(gray_img, lava_mask_threshold)
            if mode < 200:
                filled_mask = cv2.add(filled_mask, temp_filled_mask)

    lava_binary = np.zeros_like(input_image)
    lava_binary[filled_mask > 0] = [255, 255, 255]

    return lava_binary


def solution(image_path):
    image = cv2.imread(image_path)
    lava_output_image = get_lava_region(image)
    return lava_output_image
