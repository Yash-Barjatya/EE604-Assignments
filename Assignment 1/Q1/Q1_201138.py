import cv2
import numpy as np

# Usage
def generate_reference_image():
    """Function to generate the Indian flag pattern which will be used as reference.
    
    The function creats an image at a larger size (1200x1200), drawing the shapes, and then resizing it down to the original size (600x600),to apply an anti-aliasing technique. The resizing operation averages the colors of nearby pixels, which results in smoother edges and curves and thus improves the accuracy of reference image generated

    Args:
        None

    Returns:
        The reference image
    """
    # Create a blank 1200x1200 image (twice as large as original required reference size)
    reference_image = np.zeros((1200,1200,3), np.uint8)

    # Define the colors (in BGR format)
    saffron = (51, 153, 255)
    white = (255, 255, 255)
    green = (0, 128, 0)
    # blue = (255, 10, 10)#98.40% accuracy ()
    blue = (255, 0, 0)#98.39% accuracy(25.583681292840566)

    # Draw the saffron stripe
    reference_image[0:400, :] = saffron

    # Draw the white stripe
    reference_image[400:799, :] = white

    # Draw the green stripe
    reference_image[800:1200, :] = green

    # Define the center of the circle
    center_coordinates = (599, 599)

    # Define the radius of the circle
    radius = 199

    # Define the thickness of the circle
    thickness = 4

    # Draw the circle

    y, x = np.ogrid[:1200, :1200]

    # Draw the circle's perimeter by creating a mask for the desired thickness #Accuracy=98.48%(25.606327610770418)
    thickness_mask = ((x - center_coordinates[0]) ** 2 + (y - center_coordinates[1]) ** 2 <= ((radius+thickness/2) ** 2 )) & ((x - center_coordinates[0]) ** 2 + (y - center_coordinates[1]) ** 2 >= ((radius -thickness/2) ** 2))

    # Use the mask to set pixel values within the circle to blue
    reference_image[thickness_mask] = blue

    # Define the thickness of the spokes
    spoke_thickness = 2

    # Draw the spokes on the white stripe
    for angle in range(0, 360, 15):
        end_point = (int(center_coordinates[0] + (radius) * np.cos(np.radians(angle))), 
                    int(center_coordinates[1] + (radius) * np.sin(np.radians(angle))))
        reference_image = cv2.line(reference_image, center_coordinates, end_point, blue, spoke_thickness)

    # Resize down to 600x600
    reference_image = cv2.resize(reference_image, (600,600), interpolation=cv2.INTER_LINEAR)

    return reference_image


def viewImage(image):
    """Function to view an image.

    Args:
        the image to be viewed, provided as an array and not the image path

    Returns:
        returns nothing but lets you view the image
    """
    # Display the  image
    cv2.imshow('Image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_closest_border(image, color, threshold=128):
    """Function to find to which of the four boundaries (top,left,right or bottom) are the saffron pixels on an average closest to.

    Args:
       the image to be examined, provided as an array and not the image path
       the color whose closest border is to be found

    Returns:
        The reference image
    """
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert the color to a NumPy array
    color = np.array(color)

    # Calculate the mask for the specified color
    mask = cv2.inRange(image, color, color)

    # Find the coordinates of the specified color pixels using the mask
    color_pixels = np.column_stack(np.where(mask > 0))
    
    # Initialize distances with a large value
    distances = {
        "top": float('inf'),
        "bottom": float('inf'),
        "left": float('inf'),
        "right": float('inf')
    }
    # Get image dimensions
    height, width = gray_image.shape

    # Calculate distances to each border
    if len(color_pixels) > 0:
        distances = {
            "top": np.mean(color_pixels[:, 0]),
            "bottom": height - np.mean(color_pixels[:, 0]),
            "left": np.mean(color_pixels[:, 1]),
            "right": width - np.mean(color_pixels[:, 1])
        }

    # Find the closest border and the corresponding distance
    closest_border,min_distance = min(distances.items(), key=lambda x: x[1])

    return closest_border, min_distance

def solution(image_path):
    image= cv2.imread(image_path)
    ######################################################################
    ######################################################################
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################

    # define saffron color
    saffron =[51,153,255]

    # define green color
    green =[0,128,0]

    # find the border to which the saffron color is closest to and also that min distance
    saffron_closest_border,saffron_min_dist = find_closest_border(image,saffron)

    # find the border to which the green color is closest to and also that min distance
    green_closest_border,green_min_dist = find_closest_border(image,green)

    #find the min of these two min distance and return the border accordingly
    if(saffron_min_dist<=green_min_dist):
        closest_border =saffron_closest_border
    else:
        if(green_closest_border=="top"):
            closest_border="bottom"
        elif(green_closest_border=="bottom"):
            closest_border="top"
        elif(green_closest_border=="left"):
            closest_border="right"
        else:
            closest_border="left"

    # Define rotation angles for each border
    rotation_angles = {
        "top": 0,
        "bottom": 180,
        "left": 90,
        "right": 270
    }

    # Rotate the reference image based on the closest border
    reference_image = generate_reference_image()
    rotation_angle = rotation_angles.get(closest_border, 0)
    rotated_reference_image = np.rot90(reference_image, rotation_angle // 90)
    output_image = rotated_reference_image

    ######################################################################

    return output_image
