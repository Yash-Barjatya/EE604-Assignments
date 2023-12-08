import cv2
import numpy as np

def correct_upside_down(image):
    """Checks if the orientation of image is upside or upside down.

    Args:
        image: The image pixel array.

    Returns:
        A properly orientated image that is always upright
    """
    # Check if the image is grayscale
    if len(image.shape) == 2:
        gray = image
    else:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize the image
    _, bin_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Find the first black pixel for each column from the top
    top_black_pixels = []
    for col in range(bin_img.shape[1]):
        for row in range(bin_img.shape[0]):
            if bin_img[row, col] == 0:  # Black pixel found
                top_black_pixels.append(row)
                break

    # Find the first black pixel for each column from the bottom
    bottom_black_pixels = []
    for col in range(bin_img.shape[1]):
        for row in range(bin_img.shape[0] - 1, -1, -1):
            if bin_img[row, col] == 0:  # Black pixel found
                bottom_black_pixels.append(bin_img.shape[0] - 1 - row)  # Calculate distance from bottom
                break

    # Calculate the standard deviation of top and bottom black pixel positions
    top_std = np.std(top_black_pixels)
    bottom_std = np.std(bottom_black_pixels)

    # Determine if the image is upside down based on the std values
    if top_std > bottom_std:
        # Rotate the image by 180 degrees
        rotated_img = cv2.rotate(image, cv2.ROTATE_180)
        return rotated_img # Image is upside down, and rotated image is provided
    else:
        return image  # Image is in the correct orientation
    
def solution(image_path):
    
    # Read the image and convert it to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Binarize the image- This code threshold the grayscale image (img) to convert it into a binary image. The cv2.threshold function takes img as input and applies a binary threshold. Pixels with intensity greater than or equal to 128 are set to 255 (white), and pixels with intensity less than 128 are set to 0 (black). 

    _, bin_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Add a white border to the image to prevent cropping during rotation
    h, w = img.shape[:2]
    diagonal_size = int(np.sqrt(h**2 + w**2))
    border_size= diagonal_size-w
    bin_img_with_padding = cv2.copyMakeBorder(bin_img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=255)
    
    # Function to find the score for a given angle based on projection profile method
    def find_projection_score(arr, angle):
    #the below code rotate the image by the specified angle using affine transformation
        data = cv2.warpAffine(arr, cv2.getRotationMatrix2D((arr.shape[1] / 2, arr.shape[0] / 2), angle, 1), (arr.shape[1], arr.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    #The below line calculates the sum of pixel values along each row of the rotated image data. This effectively computes a histogram of the pixel intensities along the vertical axis.
        histogram = np.sum(data, axis=1)

    #This line calculates a score based on the histogram It computes the squared differences between adjacent histogram values and sums them up. This score represents how much variation there is in pixel intensities along the vertical axis of the rotated image.
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return  score
    
    #These lines define parameters for a range of angles (angles) over which the rotation will be tested. delta determines the step size between angles, and limit sets the range from -180 to 180 degrees.
    delta = 1
    limit = 180
    
    # calculate the score for all angles
    best_score = 0
    for angle in range(-limit,limit+delta,delta):
        score = find_projection_score(bin_img_with_padding, angle)
        # scores.append(score)
        if(best_score<score):
            best_score=score
            best_angle=angle
    
    # Correct skew
    rotated_img = cv2.warpAffine(bin_img_with_padding, cv2.getRotationMatrix2D((bin_img_with_padding.shape[1] / 2, bin_img_with_padding.shape[0] / 2), best_angle, 1), (bin_img_with_padding.shape[1], bin_img_with_padding.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    # if the rotated image is upside down, then correct its orientation 
    output_img=correct_upside_down(rotated_img)
    
    # Save the corrected image
    # cv2.imwrite('skew_corrected.jpg', output_img)
    return output_img