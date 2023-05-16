import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    def clamp(x, y):
        if x < 0 or x >= Hi or y < 0 or y >= Wi:
            return 0
        return image[x, y]
    for i in range(Hi):
        for j in range(Wi):
            for k in range(Hk):
                for l in range(Wk):
                    out[i, j] += clamp(i - k + Hk//2, j - l + Wk//2) * kernel[k, l]
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)))
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    padded = zero_pad(image, Hk//2, Wk//2)
    flipped = np.flip(kernel)
    # use vectorize to speed up... maybe
    f = np.vectorize(lambda i, j: np.sum(padded[i:(i+Hk), j:(j+Wk)] * flipped))
    out = f(np.arange(Hi)[:, np.newaxis], np.arange(Wi))
    # for i in range(Hi):
    #     for j in range(Wi):
    #         out[i, j] += np.sum(padded[i:(i+Hk), j:(j+Wk)] * flipped)
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    out = None
    ### YOUR CODE HERE
    out = conv_fast(f, np.flip(g))
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    out = cross_correlation(f, g - np.mean(g))
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    padded = zero_pad(f, Hg//2, Wg//2)
    g_mean = np.mean(g)
    g2 = g - g_mean
    g_std = np.std(g)
    def calculate(i, j):
        f_sub = padded[i:(i+Hg), j:(j+Wg)]
        return np.sum((f_sub - np.mean(f_sub)) * g2) / (np.std(f_sub) * g_std)
    f = np.vectorize(calculate)
    out = f(np.arange(Hf)[:, np.newaxis], np.arange(Wf))
    # for i in range(Hf):
    #     for j in range(Wf):
    #         f_sub = padded[i:(i+Hg), j:(j+Wg)]
    #         out[i, j] = np.sum((f_sub - np.mean(f_sub)) * (g - np.mean(g))) / (np.std(f_sub) * np.std(g))
    
    # out = zero_mean_cross_correlation(f, g) / np.std(g)
    # for i in range(Hf):
    #     for j in range(Wf):
    #         f_sub = f[i:(i+Hg), j:(j+Wg)]
    #         out[i, j] /= np.std(f_sub)
    ### END YOUR CODE

    return out
