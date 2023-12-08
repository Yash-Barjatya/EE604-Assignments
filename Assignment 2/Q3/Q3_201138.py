import cv2
import numpy as np


def draw_horizontal_line(thresholded_image):
    """Draws a horizontal line at the bottom of the given thresholded image.

    Args:
        thresholded_image: A binary image where white pixels represent the foreground
        and black pixels represent the background.

    Returns:
        The row index of the horizontal line.
    """
    for row in range(thresholded_image.shape[0] - 1, 0, -1):
        startingBlackPixellack_pixel = np.argmax(
            thresholded_image[row, :] == 0)
        last_black_pixel = thresholded_image.shape[1] - \
            np.argmax(thresholded_image[row, ::-1] == 0) - 1
        diff = last_black_pixel - startingBlackPixellack_pixel
        if diff < 120:
            return row - 5


def process_image(img):
    """Adds a border to the given image.

    Args:
        img: A grayscale image.

    Returns:
        A grayscale image with a border of 50 pixels on all sides.
    """
    border_size = 50
    height, width = img.shape
    img_with_border = np.ones(
        (height + 2 * border_size, width + 2 * border_size), dtype=np.uint8) * 255
    img_with_border[border_size: border_size + height,
                    border_size: border_size + width] = img
    return img_with_border


def find_minima(arr):
    """Finds all minima in the given array.

    Args:
        arr: A numpy array.

    Returns:
        A list of all minima in the array.
    """
    min_values = []
    for i in range(4, len(arr) - 4):
        curr_val = arr[i][0]
        if all(prev[0] >= curr_val for prev in arr[i - 4: i]) and all(
            next[0] >= curr_val for next in arr[i + 1: i + 5]
        ):
            if not min_values or min_values[-1] != curr_val:
                min_values.append(curr_val)
    return min_values


def find_maxima(arr, width):
    """Finds all maxima in the given array.

    Args:
        arr: A numpy array.
        width: The width of the array.

    Returns:
        A list of all maxima in the array.
    """
    max_values_idx = [0]
    for i in range(4, len(arr) - 4):
        curr_val = arr[i][0]
        if all(prev[0] <= curr_val for prev in arr[i - 4:i]) and all(next[0] <= curr_val for next in arr[i + 1:i + 5]):
            if not max_values_idx or max_values_idx[-1] != arr[i][1]:
                max_values_idx.append(arr[i][1])
    max_values_idx.append(width - 1)
    return max_values_idx


def analyze_distribution(cropped_img):
    """Analyzes the distribution of pixels in the given cropped image.

    Args:
        cropped_img: A grayscale image.

    Returns:
        The percentage of pixels in the image that are white.
    """
    cropped_img[cropped_img <= 40] = 255
    cropped_img[(cropped_img > 40) & (cropped_img <= 90)] = 0
    cropped_img[cropped_img > 90] = 255
    r, c = cropped_img.shape
    minRow, maxRow, minCol, maxCol = r + 1, 0, c + 1, 0

    for i in range(r):
        for j in range(c):
            if cropped_img[i, j] != 255:
                minRow = min(minRow, i)
                maxRow = max(i, maxRow)
                minCol = min(j, minCol)
                maxCol = max(j, maxCol)

    cropped_img = cropped_img[max(0, minRow): min(
        r, maxRow), max(0, minCol): min(r, maxCol)]
    r, c = cropped_img.shape
    cnt = np.sum(cropped_img == 0)
    return float(cnt / (r * c))


def generateArray(height, width, arr, arr3, img):
    """Generates two arrays, `arr` and `arr3`, from the given image.

    Args:
        height: The height of the image.
        width: The width of the image.
        arr: A list to store the first array.
        arr3: A list to store the second array.
        img: A grayscale image.

    Returns:
        The minimum value in the first array.
    """
    for i in range(width):
        flag = True
        for j in range(height):
            if img[j][i] != 255:
                arr.append(j)
                flag = False
                break
        if flag:
            arr.append(height)
    startingBlackPixel = 0
    for i in range(height):
        for j in range(width):
            if img[height - 1 - i][i] != 255:
                startingBlackPixel = height - 1 - i
                break
    startingBlackPixel -= 5
    st = 0
    for i in range(width):
        if img[startingBlackPixel][i] != 255:
            st = i
            break
    end = 0
    for i in range(width):
        if img[startingBlackPixel][width - 1 - i] != 255:
            end = width - 1 - i
            break
    mid = int((st + end) / 2)
    min_val = 1e9
    for i in range(mid - 10, mid + 10):
        min_val = min(arr[i], min_val)
    j = 0
    arr3.append([arr[0], 0])
    for i in range(len(arr)):
        if (arr3[j][0] != arr[i]):
            arr3.append([arr[i], i])
            j += 1
    return min_val


def solution(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = process_image(img)
    img2 = img
    img = cv2.GaussianBlur(img, (13, 13), 0)
    img[img <= 240] = 0
    img[img > 240] = 255
    height, width = img.shape
    arr = []
    arr3 = []
    min_val = generateArray(height, width, arr, arr3, img)
    min_values = find_minima(arr3)
    max_values_idx = find_maxima(arr3, width)
    left_head = 0
    right_head = 0
    for i in range(len(min_values)):
        if min_values[i] == min_val:
            left_head = i
            right_head = len(min_values) - i - 1
            break

    if left_head == 4 and right_head == 5:
        row_idx = draw_horizontal_line(img2)
        distribution = []

        for k in range(len(max_values_idx) - 1):
            cropped_img = img2[0:row_idx,
                               max_values_idx[k]:max_values_idx[k + 1]]

            distribution.append(analyze_distribution(cropped_img))

        distribution = np.array(distribution)
        mean_distribution, std_distribution = np.mean(
            distribution), np.std(distribution)

        for d in distribution:
            if d < mean_distribution - (2) * std_distribution or d > mean_distribution + (2) * std_distribution:
                return "fake"

        return "real"

    return "fake"
