import numpy as np
import cv2 as cv


def identify_lane_borders(image):
    """
    Функция для определения границ дорог по координате х;
    ------------------------------------------------------------
    :param image: трехмерный np.ndarray;

    ------------------------------------------------------------
    :return: двумерный np.ndarray, строка - координаты
    начала и конца дороги по х;
    ------------------------------------------------------------
    """

    lane_start, borders = None, []

    for i in range(image.shape[1]):

        if image[0, i] != 255 and lane_start is None:
            lane_start = i

        if image[0, i] != 0 and lane_start is not None:
            border = np.array([lane_start, i - 1], dtype=np.uint32)
            borders.append(border)

            lane_start = None

    return np.array(borders)


def find_road_number(image: np.ndarray) -> int:
    """
    Найти номер дороги, на которой нет препятсвия в конце пути.

    :param image: исходное изображение
    :return: номер дороги, на которой нет препятсвия на дороге
    """

    image_gray = cv.cvtColor(image, code=cv.COLOR_RGB2GRAY)

    lanes_low, lanes_high = 230, 255
    lanes = cv.inRange(image_gray, lanes_low, lanes_high)

    car_low, car_high = np.array([40, 100, 250]), np.array([60, 121, 255])
    car = cv.inRange(image, car_low, car_high)

    barriers_low, barriers_high = np.array([250, 30, 0]), np.array([255, 50, 10])
    barriers = cv.inRange(image, barriers_low, barriers_high)

    borders_lane = identify_lane_borders(lanes)
    borders_car = np.where(car == 255)[1]
    borders_barriers = np.where(barriers == 255)[1]

    lane_car, lane_free = None, None

    for i, border in enumerate(borders_lane):

        left, right = border

        ind_car = np.where((borders_car > left) & (borders_car < right))[0]
        ind_free = np.where((borders_barriers > left) & (borders_barriers < right))[0]

        if len(ind_free) == 0:
            lane_free = i

        if len(ind_car) > 0:
            line_car = i

    return lane_free
