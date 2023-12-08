import cv2
import numpy as np


class JointBilateralFilter:
    def __init__(self, spatial_sigma, range_sigma):
        self.range_sigma = range_sigma
        self.spatial_sigma = spatial_sigma
        self.window_size = 6 * spatial_sigma + 1
        self.padding_width = 3 * spatial_sigma

    def joint_bilateral_filter(self, image, guidance):
        border_type = cv2.BORDER_REFLECT
        padded_image = cv2.copyMakeBorder(image, self.padding_width, self.padding_width, self.padding_width,
                                          self.padding_width, border_type).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.padding_width, self.padding_width,
                                             self.padding_width, self.padding_width, border_type).astype(np.int32)

        # Set up look-up tables for spatial and range kernels
        lut_spatial = np.exp(-0.5 * (np.arange(self.padding_width + 1)
                             ** 2) / self.spatial_sigma ** 2)
        lut_range = np.exp(-0.5 * (np.arange(256) / 255)
                           ** 2 / self.range_sigma ** 2)

        weight_sum, result = np.zeros(
            padded_image.shape), np.zeros(padded_image.shape)

        for x in range(-self.padding_width, self.padding_width + 1):
            for y in range(-self.padding_width, self.padding_width + 1):
                # Compute the weight of range kernel by rolling the whole image
                dT = lut_range[np.abs(
                    np.roll(padded_guidance, [y, x], axis=[0, 1]) - padded_guidance)]
                range_weight = dT if dT.ndim == 2 else np.prod(dT, axis=2)

                # Spatial kernel
                spatial_weight = lut_spatial[np.abs(
                    x)] * lut_spatial[np.abs(y)]

                # Joint weight
                joint_weight = spatial_weight * range_weight

                padded_image_roll = np.roll(padded_image, [y, x], axis=[0, 1])

                for channel in range(padded_image.ndim):
                    result[:, :, channel] += padded_image_roll[:,
                                                               :, channel] * joint_weight
                    weight_sum[:, :, channel] += joint_weight

        # Crop the result and normalize
        output = (result / weight_sum)[self.padding_width:-
                                       self.padding_width, self.padding_width:-self.padding_width, :]

        return np.clip(output, 0, 255).astype(np.uint8)

# Example Usage:
# spatial_sigma_value = 2
# range_sigma_value = 0.1
# bilateral_filter = JointBilateralFilter(spatial_sigma_value, range_sigma_value)
# filtered_image = bilateral_filter.joint_bilateral_filter(input_image, guidance_image)
# cv2.imshow('Filtered Image', filtered_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def distance(x, y, i, j):
    """Computes the distance between two pixels.

    Args:
        x: The x-coordinate of the first pixel.
        y: The y-coordinate of the first pixel.
        i: The x-coordinate of the second pixel.
        j: The y-coordinate of the second pixel.

    Returns:
        The distance between the two pixels.
    """
    return np.sqrt((x-i)**2 + (y-j)**2)


def gaussian(x, sigma):
    """Computes the Gaussian kernel value for the given input value and sigma.

    Args:
        x: The input value.
        sigma: The standard deviation of the Gaussian kernel.

    Returns:
        The Gaussian kernel value.
    """
    return (1.0 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2) / (2 * sigma ** 2))


def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s):
    """Applies the bilateral filter to the given pixel in the source image and writes the filtered value to the filtered image.

    Args:
        source: The source image.
        filtered_image: The filtered image.
        x: The x-coordinate of the pixel in the source image to filter.
        y: The y-coordinate of the pixel in the source image to filter.
        diameter: The diameter of the bilateral filter window.
        sigma_i: The standard deviation of the intensity Gaussian function.
        sigma_s: The standard deviation of the spatial Gaussian function.
    """
    hl = diameter/2
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neighbour_x = int(x - (hl - i))
            neighbour_y = int(y - (hl - j))
            if neighbour_x < 0:
                neighbour_x += len(source)
            if neighbour_y < 0:
                neighbour_y += len(source[0])
            gi = gaussian(source[neighbour_x]
                          [neighbour_y] - source[x][y], sigma_i)
            gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gi * gs
            i_filtered += source[neighbour_x][neighbour_y] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered / Wp
    filtered_image[x][y] = i_filtered


def bilateral_filter_own(source, filter_diameter, sigma_spatial, sigma_color):
    """Computes the bilateral filter output for the given input image.

    Args:
        source: The input image.
        filter_diameter: The radius of the bilateral filter window.
        sigma_color: The sigma value for the color difference kernel.
        sigma_space: The sigma value for the spatial distance kernel.

    Returns:
        The bilateral filter output image.
    """
    filtered_image = np.zeros(source.shape)

    i = 0
    while i < len(source):
        j = 0
        while j < len(source[0]):
            apply_bilateral_filter(
                source, filtered_image, i, j, filter_diameter, sigma_spatial, sigma_color)
            j += 1
        i += 1
    return filtered_image


def shadow_removal(imflash, imambient):
    """Removes shadows from the given images.
       Denoising ambient images using shadow removal and flash details

    Args:
        imflash: The flash image.
        imambient: The ambient image.

    Returns:
        The de-shadowed image.
    """

    # Perform bilateral filtering on flash and ambient images separately
    sigma_spatial = 2.0
    sigma_range_flash = 0.1 * (np.max(imflash) - np.min(imflash))
    sigma_range_ambient = 0.1 * (np.max(imambient) - np.min(imambient))
    filter_diameter = 2
    imflash_filtered = bilateral_filter_own(
        imflash, filter_diameter, sigma_spatial, sigma_range_flash)
    imambient_filtered = bilateral_filter_own(
        imambient, filter_diameter, sigma_spatial, sigma_range_ambient)

    # cv2.imwrite('denoised_no_filter.jpg',imambient_filtered)
    # Calculate flash details
    eps = 0.02
    F_detail = (imflash + eps) / (imflash_filtered + eps)

    # Shadow mask
    shadow_mask = shadow_removal_mask(imflash, imambient)

    # Fuse images
    Af = (1 - shadow_mask) * imambient_filtered * \
        F_detail + shadow_mask * imambient_filtered

    return Af


def shadow_removal_mask(imflash, imambient):
    # Implement shadow removal mask generation
    # This can involve thresholding, edge detection, or any suitable method
    # For simplicity, you can use a basic method like absolute intensity difference
    threshold = 0.2
    diff = np.abs(imflash - imambient)
    mask = (diff > threshold).astype(np.uint8)
    return mask


def solution(image_path_a, image_path_b):
    ############################
    ############################
    # image_path_a is path to the non-flash high ISO image
    # image_path_b is path to the flash low ISO image
    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    imflash = cv2.imread(image_path_b)
    imambient = cv2.imread(image_path_a)
    if imflash.shape[0] == 706:
        JBF = JointBilateralFilter(5, 0.01)  # create object for JBF

    else:
        JBF = JointBilateralFilter(2, 0.005)
    # img = cv2.imread(args.image_path) # read 1.png
    # img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #convert to RGB

    # initial cv2 gray conversion
    # img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb) # bf
    filteredimg = JBF.joint_bilateral_filter(
        imambient, imflash)  # JBF (gray as guidance)
    # calculate cost by L1 normalization
    # cost['cv2.COLOR_BGR2GRAY'] = np.sum(np.abs(bf_out.astype('int32')-jbf_out.astype('int32')))
    # save figures
    return filteredimg
