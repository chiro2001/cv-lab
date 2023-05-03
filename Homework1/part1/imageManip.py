import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    # YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    # END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def crop_image(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index we want to include in our cropped image.
        start_col (int): The starting column index we want to include in our cropped image.
        num_rows (int): Number of rows in our desired cropped image.
        num_cols (int): Number of columns in our desired cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """

    out = None

    # YOUR CODE HERE
    out = image[start_row:(start_row+num_rows),
                start_col:(start_col+num_cols), :]
    # END YOUR CODE

    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    # YOUR CODE HERE
    out = image**2 * 0.5
    # END YOUR CODE

    return out


def resize_image(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))

    # 2. Populate the `output_image` array using values from `input_image`
    #    > This should require two nested for loops!

    # YOUR CODE HERE
    for i in range(output_rows):
        for j in range(output_cols):
            output_image[i, j, :] = input_image[int(
                i*input_rows/output_rows), int(j*input_cols/output_cols), :]
    # END YOUR CODE

    # 3. Return the output image
    return output_image


def resize_image2(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))

    # 2. Populate the `output_image` array using values from `input_image`
    #    > This should require two nested for loops!

    # YOUR CODE HERE
    # 限制值在 [smallest, largest] 范围内
    def clamp(n, smallest, largest):
        return max(smallest, min(n, largest))
    # x, y 方向的比例
    dx, dy = output_rows / input_rows, output_cols / input_cols
    # 对一个输出图像的像素进行处理的时候，对应多少输入像素点
    pixel_out = int(dx + 1) * int(dy + 1)
    # 累加像素的时候的加权
    rate = ((output_cols * output_rows) /
            (input_cols * input_rows)) / pixel_out
    # 迭代输入图像
    for i in range(input_rows):
        for j in range(input_cols):
            # （实际上是滤波一下）
            for idx in range(int(dx + 1)):
                for idy in range(int(dy + 1)):
                    output_image[clamp(int(i * dx) + idx, 0, output_rows-1),
                                 clamp(int(j * dy) + idy, 0, output_cols-1), :] += \
                        np.array(input_image[i, j, :]) * rate
    # END YOUR CODE

    # 3. Return the output image
    return output_image


def rotate2d(point, theta):
    """Rotate a 2D coordinate by some angle theta.

    Args:
        point (np.ndarray): A 1D NumPy array containing two values: an x and y coordinate.
        theta (float): An theta to rotate by, in radians.

    Returns:
        np.ndarray: A 1D NumPy array containing your rotated x and y values.
    """
    assert point.shape == (2,)
    assert isinstance(theta, float)

    # Reminder: np.cos() and np.sin() will be useful here!

    # YOUR CODE HERE
    trans = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), -np.cos(theta)],
    ])
    return np.matmul(trans, point)
    # END YOUR CODE


def rotate2dc(point, theta, center):
    assert point.shape == (2,)
    assert center.shape == (2,)
    assert isinstance(theta, float)
    tx, ty = center
    cos, sin = np.cos, np.sin
    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float32)
    R = np.array([
        [cos(theta), -sin(theta), 0],
        [sin(theta), cos(theta), 0],
        [0 , 0, 1]
    ])
    S = np.eye(3)
    P = np.array([point[0], point[1], 1], dtype=np.float32).T
    M = np.matmul(T, R)
    M = np.matmul(M, S)
    M = np.matmul(M, P)
    return M[:2]
    


def rotate_image(input_image, theta):
    """Rotate an image by some angle theta.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        theta (float): Angle to rotate our image by, in radians.

    Returns:
        (np.ndarray): Rotated image, with the same shape as the input.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create an output image with the same shape as the input
    output_image = np.zeros_like(input_image)
    output_rows, output_cols = output_image.shape[:2]

    # YOUR CODE HERE
    center = np.array([input_rows / 2.0, input_cols / 2.0], dtype=np.float32)
    # center = np.array([10, 0], dtype=np.float32)
    for i in range(output_rows):
        for j in range(output_cols):
            x, y = rotate2dc(np.array([i, j], dtype=np.float32) - center, theta, center)
            x, y = int(x), int(y)
            if 0 <= x < input_rows and 0 <= y < input_cols:
                output_image[i, j, :] = input_image[x, y, :]
    # END YOUR CODE

    # 3. Return the output image
    return output_image
