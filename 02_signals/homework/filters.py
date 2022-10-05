from numpy.lib.stride_tricks import as_strided
import numpy as np


def expand_kernel(kernel):
    """
    Add one zero column if column's amount is even.
    Add one zero row if row's amount is even.

    Args:
        kernel: np.ndarray of shape(hk, wk)

    Returns: np.ndarray of expanded shape

    """

    if kernel.shape[0] % 2 == 0:
        kernel = np.vstack((kernel, np.zeros(kernel.shape[1])))

    if kernel.shape[1] % 2 == 0:
        kernel = np.hstack((kernel, np.zeros(kernel.shape[0])))

    return kernel


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (hi, wi).
        kernel: numpy array of shape (hk, wk).

    Returns:
        out: numpy array of shape (hi, wi).
    """

    kernel = expand_kernel(kernel)

    hi, wi = image.shape
    hk, wk = kernel.shape

    out = np.zeros((hi, wi))

    for i in range(hi):
        for j in range(wi):
            for k in range(hk):
                for m in range(wk):

                    image_i, image_j = i + k - hk // 2, j + m - wk // 2

                    if image_i < 0 or image_i >= hi:
                        continue

                    if image_j < 0 or image_j >= wi:
                        continue

                    value = image[image_i, image_j] * kernel[hk - k - 1, wk - m - 1]
                    out[i, j] += value

    return out


def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (h, w).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (h+2*pad_height, w+2*pad_width).
    """

    h, w = image.shape

    out = np.zeros((h + 2 * pad_height, w + 2 * pad_width))
    out[pad_height: h + pad_height, pad_width: w + pad_width] = image

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
        image: numpy array of shape (hi, wi).
        kernel: numpy array of shape (hk, wk).

    Returns:
        out: numpy array of shape (hi, wi).
    """

    ker_reversed = np.flip(kernel)
    ker_reversed = expand_kernel(ker_reversed)

    hi, wi = image.shape
    hk, wk = ker_reversed.shape

    out = np.zeros((hi, wi))
    padded = zero_pad(image, hk // 2, wk // 2)

    for i in range(hk // 2, hk // 2 + hi):
        for j in range(wk // 2, wk // 2 + wi):

            val_conv = padded[i - hk // 2: i + hk // 2 + 1, j - wk // 2: j + wk // 2 + 1]
            out[i - hk // 2, j - wk // 2] = np.sum(val_conv * ker_reversed)

    return out


def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (hi, wi).
        kernel: numpy array of shape (hk, wk).

    Returns:
        out: numpy array of shape (hi, wi).
    """

    ker_reversed = np.flip(kernel)
    ker_reversed = expand_kernel(ker_reversed)

    hi, wi = image.shape
    hk, wk = ker_reversed.shape

    padded = zero_pad(image, hk // 2, wk // 2)

    strided_shape = (hi, wi, hk, wk)
    padded_strided = as_strided(padded, strided_shape, strides=padded.strides * 2)

    out = np.sum(padded_strided * ker_reversed, axis=(2, 3))

    return out


def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    kernel = np.flip(g)
    out = conv_faster(f, kernel)

    return out


def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    kernel = np.flip(g) - g.mean()
    out = conv_faster(f, kernel)

    return out


def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

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

    kernel = expand_kernel(g)

    hi, wi = f.shape
    hk, wk = kernel.shape

    padded = zero_pad(f, hk // 2, wk // 2)

    strided_shape = (hi, wi, hk, wk)
    strided = as_strided(padded, strided_shape, strides=padded.strides * 2)

    strided_mean = strided.mean(axis=(2, 3))[:, :, np.newaxis, np.newaxis]
    strided_std = strided.std(axis=(2, 3))[:, :, np.newaxis, np.newaxis]
    strided = (strided - strided_mean) / strided_std
    kernel = (kernel - kernel.mean()) - kernel.std()

    out = np.sum(strided * kernel, axis=(2, 3))

    return out
