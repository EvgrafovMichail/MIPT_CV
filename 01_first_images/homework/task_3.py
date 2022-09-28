import numpy as np
import cv2 as cv


def rotate(image, point: tuple, angle: float) -> np.ndarray:
    """
    Повернуть изображение по часовой стрелке на угол от 0 до 360 градусов и преобразовать размер изображения.

    :param image: исходное изображение
    :param point: значение точки (x, y), вокруг которой повернуть изображение
    :param angle: угол поворота
    :return: повернутное изображение
    """
    verticies = np.array([[image.shape[1] - 1, image.shape[0] - 1],
                          [0, 0], [image.shape[1] - 1, 0],
                          [0, image.shape[0] - 1]])

    matrix = cv.getRotationMatrix2D(point, angle, scale=1.0)

    verticies_r3 = np.hstack((verticies, np.ones((verticies.shape[0], 1))))
    verticies_transformed = matrix @ verticies_r3.T

    matrix[:, 2] -= verticies_transformed.min(axis=1)
    new_shape = verticies_transformed.max(axis=1) - verticies_transformed.min(axis=1)
    new_shape = np.int64(np.ceil(new_shape))

    return cv.warpAffine(image, matrix, new_shape)


def apply_warpAffine(image, points1, points2) -> np.ndarray:
    """
    Применить афинное преобразование согласно переходу точек points1 -> points2 и
    преобразовать размер изображения.

    :param image:
    :param points1:
    :param points2:
    :return: преобразованное изображение
    """

    verticies = np.array([[image.shape[1] - 1, image.shape[0] - 1],
                          [0, 0], [image.shape[1] - 1, 0],
                          [0, image.shape[0] - 1]])

    matrix = cv.getAffineTransform(points1, points2)

    verticies_r3 = np.hstack((verticies, np.ones((verticies.shape[0], 1))))
    verticies_transformed = matrix @ verticies_r3.T

    matrix[:, 2] -= verticies_transformed.min(axis=1)
    new_shape = verticies_transformed.max(axis=1) - verticies_transformed.min(axis=1)
    new_shape = np.int64(np.ceil(new_shape))

    return cv.warpAffine(image, matrix, new_shape)
