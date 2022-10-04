import numpy as np


WHITE = np.array([255, 255, 255], dtype=np.uint8)   # RGB-код белого цвета
BLACK = np.array([0, 0, 0], dtype=np.uint8)         # RGB-код чёрного цвета


########################################################################################################################
# Класс для решения задачи; реализация волнового алгоритма;                                                            #
########################################################################################################################


class MazeGraph:
    """
    Класс реализующий волновой алгоритм
    ------------------------------------------------------------
    Инициализация:

    - image_maze - трехмерный np.ndarray размеров(N, M, 3) -
    RGB-изображения лабиринта;
    ------------------------------------------------------------

    """

    def __init__(self, image_maze):

        self.__initialize_data(image_maze)
        self.__find_route()

    def __initialize_data(self, image_maze):
        """
        Функция для инициализации полей класса
        ------------------------------------------------------------
        :param image_maze: трехмерный np.ndarray размеров(N, M, 3) -
        RGB-изображения лабиринта;

        ------------------------------------------------------------
        :return: None
        ------------------------------------------------------------
        """

        self.image = image_maze

        enter_start, enter_end = self.__identify_gap_h(0)
        exit_start, exit_end = self.__identify_gap_h(self.image.shape[0] - 1)
        wall_start, wall_end = self.__identify_gap_v(enter_start)

        self.cell_width = enter_end - enter_start
        wall_width = wall_end - wall_start

        self.step = self.cell_width + wall_width

        shape = (self.image.shape[0] // self.step,
                 self.image.shape[1] // self.step)

        self.cells_visited = np.zeros(shape=shape, dtype=np.int32)

        self.start = np.array([wall_width, enter_start], dtype=np.int32)
        self.stop = np.array([self.image.shape[0] - self.step, exit_start],
                             dtype=np.int32)

        self.queue = []
        self.route = []

    def __get_adj(self, i, j):
        """
        Функция для нахождения клеток лабиринта, смежных данной
        ------------------------------------------------------------
        :param i: целое число - номер строки изображения, верхний
        левый угол текущей клетки;
        :param j: целое число - номер столбца изображения, верхний
        левый угол текущей клетки;

        ------------------------------------------------------------
        :return: список координат верхних левых углов смежный клеток;
        ------------------------------------------------------------
        """

        adj = []

        if np.all(self.image[i + self.cell_width, j] == WHITE) and i + self.step < self.image.shape[0]:
            adj.append(np.array([i + self.step, j], dtype=np.int32))

        if np.all(self.image[i, j + self.cell_width] == WHITE) and j + self.step < self.image.shape[1]:
            adj.append(np.array([i, j + self.step], dtype=np.int32))

        if np.all(self.image[i - 1, j] == WHITE) and i - self.step > 0:
            adj.append(np.array([i - self.step, j], dtype=np.int32))

        if np.all(self.image[i, j - 1] == WHITE) and j - self.step > 0:
            adj.append(np.array([i, j - self.step], dtype=np.int32))

        return adj

    def __forward_pass(self):
        """
        Функция прямого распространения волнового алгоритма;
        Сопоставляет номер волны каждой посещённой клетки лабиринта;
        ------------------------------------------------------------

        :return: None
        ------------------------------------------------------------
        """

        i, j = self.start // self.step
        self.cells_visited[i, j] = 1

        cell_curr = None
        is_exit_found = False

        self.queue.append(self.start)

        while len(self.queue) and not is_exit_found:

            cell_curr = self.queue.pop(0)
            adjs = self.__get_adj(cell_curr[0], cell_curr[1])

            for adj in adjs:

                i_curr, j_curr = cell_curr // self.step
                i, j = adj // self.step

                if self.cells_visited[i, j] == 0:

                    self.cells_visited[i, j] = self.cells_visited[i_curr, j_curr] + 1

                    if np.all(adj == self.stop):
                        is_exit_found = True
                        break

                    self.queue.append(adj)

    def __backward_pass(self):
        """
        Функция для формирования маршрута по проставленным номерам
        волн;
        ------------------------------------------------------------
        :return: None
        ------------------------------------------------------------
        """

        i, j = self.stop // self.step

        dist_curr = self.cells_visited[i, j]
        cell_curr = self.stop

        self.route.append(self.stop)

        while dist_curr != 1:

            adjs = self.__get_adj(cell_curr[0], cell_curr[1])

            for adj in adjs:

                i, j = adj // self.step

                if self.cells_visited[i, j] == dist_curr - 1:
                    cell_curr = adj
                    dist_curr -= 1

                    break

            self.route.append(cell_curr)

    def __identify_gap_h(self, row):
        """
        Функция для определения величины разрыва однотонной
        горизонтальной линии в пикселях;
        ------------------------------------------------------------

        :param row: целое число, номер строки, содержащей исследуемую
        линию;
        ------------------------------------------------------------
        :return: номера столбцов начала и конца линии;
        ------------------------------------------------------------
        """

        color_prev = self.image[row, 0]
        gap_start, gap_end = None, None

        for i in range(1, self.image.shape[1]):

            if np.all(self.image[row, i] != color_prev):
                color_prev = self.image[row, i]
                gap_start = i
                break

        for i in range(gap_start, self.image.shape[1]):

            if np.all(self.image[row, i] != color_prev):
                gap_end = i
                break

        return gap_start, gap_end

    def __identify_gap_v(self, column):
        """
        Функция для определения величины разрыва однотонной
        вертикальной линии в пикселях;
        ------------------------------------------------------------

        :param column: целое число, номер столбца, содержащего
        исследуемую линию;
        ------------------------------------------------------------
        :return: номера строк начала и конца линии;
        ------------------------------------------------------------
        """

        color_prev = self.image[0, column]
        gap_start, gap_end = None, None

        for i in range(1, self.image.shape[0]):

            if np.all(self.image[i, column] != color_prev):
                color_prev = self.image[i, column]
                gap_start = i
                break

        for i in range(gap_start, self.image.shape[0]):

            if np.all(self.image[i, column] != color_prev):
                gap_end = i
                break

        return gap_start, gap_end

    def __find_route(self):
        """
        Функция для вычисления маршрута через лабиринт
        ------------------------------------------------------------
        :return: None;
        ------------------------------------------------------------
        """

        if len(self.route) != 0:
            return self.route

        self.__forward_pass()
        self.__backward_pass()

        self.route.reverse()

        route = np.array(self.route)
        route += self.cell_width // 2

        exit = np.array([self.image.shape[0], route[-1, 1]], dtype=np.int32)
        enter = np.array([0, route[0, 1]], dtype=np.int32)

        self.route = np.vstack((enter, route, exit))


########################################################################################################################
# Функция для восстановления маршрута;                                                                                 #
########################################################################################################################


def find_way_from_maze(image: np.ndarray) -> tuple:
    """
    Найти путь через лабиринт.

    :param image: изображение лабиринта
    :return: координаты пути из лабиринта в виде (x, y), где x и y - это массивы координат
    """

    route = MazeGraph(image).route

    coords = np.array(route[0], dtype=np.int32)

    for i in range(route.shape[0] - 1):

        start, stop = route[i], route[i + 1]

        x, y = None, None

        if start[0] == stop[0]:

            step = 1 if start[1] < stop[1] else -1
            x = np.ones(np.abs(stop[1] - start[1]), dtype=np.int32) * stop[0]
            y = np.arange(start[1], stop[1], step, dtype=np.int32)

        else:

            step = 1 if start[0] < stop[0] else -1
            x = np.arange(start[0], stop[0], step, dtype=np.int32)
            y = np.ones(np.abs(stop[0] - start[0]), dtype=np.int32) * stop[1]

        points = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
        coords = np.vstack((coords, points))

    return coords[:, 0], coords[:, 1]
