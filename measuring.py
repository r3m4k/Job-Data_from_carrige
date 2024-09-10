# -*- coding: utf-8 -*-

import os
from sys import exit
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class Measuring:
    def __init__(self, dir, filter_type, end_height, testFlag):
        """
        Класс, предназначенный для "стягивания" графиков в определённых точка

        :param dir: Директория расположения сохранённых файлов, относительно расположения программы.
        :param filter_type: Используемый фильтр.
                            Возможные варианты: average_filter - фильтр усреднения,
                                                median_filter - медианный фильтр.
        :param end_height: Высота в конце проезда, относительно его начала.
        :param testFlag: Флаг тестового режима. Если True, то по ходу выполнения программы
                         будут появляться вспомогательные изображения, поясняющие выполняемые шаги.
        """

        self.dir = dir
        self.filter_type = filter_type
        self.end_height = end_height
        self.testFlag = testFlag

        # Функции используемого фильтра
        self.filters = {'average_filter': self.average_filter,
                        'median_filter': self.median_filter}

        self.files = self.get_files()       # Список CSV файлов
        self.coordinates = []               # Список массивов координат, прочитанный из files. После self.interpolation станет массивом координат
        self.marked_coordinates = []        # Список массивов координат, прочитанный из files, которые отмечены отметкой Piket.
                                            # После self.get_common_marked_coordinates он станет массивом отмеченных координат. По этим координатам будет происходить "стяжка"
        self.std_array = np.ndarray         # Массив среднеквадратичных отклонений

        self.step = 0.22553                 # Шаг с которым будет проводиться интерполяция

    ############# Получение списка файлов #############

    def get_files(self):
        """
        Получение списка CSV файлов, которые находятся в self.dir
        :return: список CSV файлов
        """
        files = []
        with os.scandir(self.dir) as items:
            for item in items:
                if ".csv" in item.name and item.is_file():
                    files.append(item.name)

        if len(files) == 0:
            print()
            print('Не обнаружено ни одного CSV файла')
            exit(1)

        return files

    ############# Чтение данных #############

    def reading_values(self):
        """
        Чтение данных с последующим сохранением координат в self.coordinates, а отмеченных координат в self.marked_coordinates.
        :return Курс, крен и тангаж в виде списка массивов, прочитанные в файлах self.files.
        """

        course = []
        roll = []
        pitch = []
        track_width = []

        for fileIndex in range(len(self.files)):

            try:
                course.append(self.clearing(np.loadtxt(f'{self.dir}/{self.files[fileIndex]}', dtype=str, skiprows=1, unpack=True, encoding="UTF-8")[6]) * np.pi / 180)      # В радианах
                roll.append(self.clearing(np.loadtxt(f'{self.dir}/{self.files[fileIndex]}', dtype=str, skiprows=1, unpack=True, encoding="UTF-8")[7]) * np.pi / 180)
                pitch.append(self.clearing(np.loadtxt(f'{self.dir}/{self.files[fileIndex]}', dtype=str, skiprows=1, unpack=True, encoding="UTF-8")[8]) * np.pi / 180)
                track_width.append(self.clearing(np.loadtxt(f'{self.dir}/{self.files[fileIndex]}', dtype=str, skiprows=1, unpack=True, encoding="UTF-8")[4]) / 1000)        # В метрах

                self.coordinates.append(self.clearing(np.loadtxt(f'{self.dir}/{self.files[fileIndex]}', dtype=str, skiprows=1, unpack=True, encoding="UTF-8")[0]))

                marks = np.loadtxt(f'{self.dir}/{self.files[fileIndex]}', dtype=str, skiprows=1, unpack=True, encoding="UTF-8")[3]
                marked_coordinates = np.zeros(0)
                for markIndex in range(len(marks)):
                    if marks[markIndex] == 'Piket':
                        marked_coordinates = np.append(marked_coordinates, self.coordinates[fileIndex][markIndex])

                self.marked_coordinates.append(marked_coordinates)

            except Exception:      # Ошибка, возникающая, если в self.dir присутствуют другие CSV файлы, который не подходят под шаблон CSV файла с измерениями
                print(f'Ошибка чтения {self.files[fileIndex]}')

        return course, roll, pitch, track_width

    @staticmethod
    def clearing(column):
        """
        Функция, которая заменяет "," на "." в column для проведения дальнейших математических операций.
        :param column: Массив элементов, содержащих "," вместо "." из-за чтения CSV файла.
        :return: np.array с типом элементов float, а не str
        """
        arr = np.zeros(len(column))
        for i in range(len(column)):
            arr[i] = column[i].replace(",", ".")

        return arr.astype(float)

    ############# Интерполяция с определённым шагом #############

    def interpolation(self, course=None, roll=None, pitch=None, track_width=None):
        """
        Получение значений курса, крена, тангажа и ширины в одних и тех же координатах с шагом self.step с диапазоном от наименьшей минимальной координаты до наибольшей максимальной координаты.
        После выполнения функции список массивов координат self.coordinates станет массивом координат с шагом self.step с таким же диапазоном.
        :return Значения курса, крена, тангажа и ширины колеи в одних и тех же координатах
        """

        for fileIndex in range(len(self.files)):
            reverseFlag = False if self.coordinates[fileIndex][0] < self.coordinates[fileIndex][-1] else True       # Флаг развёрнутого файла
            # Тк в файлах *_out.csv координаты идут по возрастанию, а в *_in.csv по убыванию

            min_coordinate = self.coordinates[fileIndex][0] if not reverseFlag else self.coordinates[fileIndex][-1]
            max_coordinate = self.coordinates[fileIndex][-1] if not reverseFlag else self.coordinates[fileIndex][0]
            max_coordinate -= self.step     # Необходимо, чтобы справа от coord всегда была точка из self.coordinates[fileIndex].
                                            # Это необходимо для корректной работы self.line_coefficients.

            coord_array = np.zeros(0)

            if course:
                coord = (min_coordinate // self.step + 1) * self.step       # Таким образом мы гарантируем, что coord будет кратен self.step и больше
                                                                            # min_coordinate, что необходимо для корректной работы self.line_coefficients,
                                                                            # тк coord должна всегда находится между двумя точками из self.coordinates[fileIndex]
                coord_array = np.zeros(0)
                file_course_array = np.zeros(0)

                while coord < max_coordinate:
                    line = self.line_coefficients(coord, self.coordinates[fileIndex], course[fileIndex])
                    coord_array = np.append(coord_array, coord)
                    file_course_array = np.append(file_course_array, line[0] * coord + line[1])
                    coord += self.step

                course[fileIndex] = file_course_array

            if roll:
                coord = (min_coordinate // self.step + 1) * self.step

                coord_array = np.zeros(0)
                file_roll_array = np.zeros(0)

                while coord < max_coordinate:
                    line = self.line_coefficients(coord, self.coordinates[fileIndex], roll[fileIndex])
                    coord_array = np.append(coord_array, coord)
                    file_roll_array = np.append(file_roll_array, line[0] * coord + line[1])
                    coord += self.step

                roll[fileIndex] = file_roll_array

            if pitch:
                coord = (min_coordinate // self.step + 1) * self.step

                coord_array = np.zeros(0)
                file_pitch_array = np.zeros(0)

                while coord < max_coordinate:
                    line = self.line_coefficients(coord, self.coordinates[fileIndex], pitch[fileIndex])
                    coord_array = np.append(coord_array, coord)
                    file_pitch_array = np.append(file_pitch_array, line[0] * coord + line[1])
                    coord += self.step

                pitch[fileIndex] = file_pitch_array

            if track_width:
                coord = (min_coordinate // self.step + 1) * self.step

                coord_array = np.zeros(0)
                file_track_width_array = np.zeros(0)

                while coord < max_coordinate:
                    line = self.line_coefficients(coord, self.coordinates[fileIndex], track_width[fileIndex])
                    coord_array = np.append(coord_array, coord)
                    file_track_width_array = np.append(file_track_width_array, line[0] * coord + line[1])
                    coord += self.step

                track_width[fileIndex] = file_track_width_array

            self.coordinates[fileIndex] = coord_array

        return course, roll, pitch, track_width

    ############# Подгон графиков к конечной высоте ##############

    def fitting_to_finalHeight(self, data):
        # Найдём файл с данными, записанными до максимальной координаты
        max_coordinates = []
        for dataIndex in range(len(data)):
            max_coordinates.append(self.coordinates[dataIndex][-1])
        max_length = max(max_coordinates)

        for dataIndex in range(len(data)):
            if data[dataIndex][-1] == max_length:
                # Опустим график самого длинного файла
                line_coefficient = (data[dataIndex][-1] - self.end_height) / self.coordinates[dataIndex][-1]
                for i in range(len(data[dataIndex])):
                    data[dataIndex][i] -= line_coefficient * (self.coordinates[dataIndex][i] - self.coordinates[dataIndex][0])
            else:
                # Тк в файлах меньшей длины нет данных, соответствующих максимальной координате, то опустим данные в таких фалах так,
                # чтобы на одной линии лежали точки (0, 0), (self.coordinate[index][-1], data[index][-1]), (max_length, self.end_height)
                k = data[dataIndex][-1] / self.coordinates[dataIndex][-1]
                end_fitting_data = k * max_length
                line_coefficient = (end_fitting_data - self.end_height) / max_length
                for i in range(len(data[dataIndex])):
                    data[dataIndex][i] -= line_coefficient * (self.coordinates[dataIndex][i] - self.coordinates[dataIndex][0])

    ############# Дополнение данных до общего начала и общего конца #############

    def filling_beginning(self, source_data, vertical_profileFlag=False, coordinate_changes_allowed=True):
        """
        Заполнение source_data до минимальной координаты, если файл записан не полностью.
        Например, есть три файла, записанные с 0, 10, 15 метров до 200 метров соответственно.
        Тогда по выполнению данной функции все три файла будут заполнены от 0 до 200 метров таким образом:
        Для крена:
            1. К данным со второго файла прибавляется величина из первого, соответсвующая начальной координате второго файла
               (т.о. график данных со второго файла поднимется и совпадёт с первым в точке начала отсчёта второго файла)
            2. Данные второго файла дополняются до 0 метров значениями первого
            3. К данным третьего файла прибавляется величина, равная среднему значению величин первого и второго файлов в координате, соответствующей началу отсчёта третьего файла.
            4. Данные третьего файла дополняются до 0 метров средними значениями первого и второго в диапазоне от 0 м до 15 м
        Для курса:
               Аналогично крену, но без 1-го пункта.
        Аналогично с любым количеством файлов.

        Важно!  Если данные в файле записаны от 100 метров (больше половины максимальной длины), то такой файл считается нерепрезентативны и удаляется.

        :param source_data: Исходные данные.
        :param vertical_profileFlag: Флаг вертикального профиля. Используется для условия подъёма графиков данных.
        :param coordinate_changes_allowed: Тк может понадобиться дополнить сразу несколько величин, то этот флаг необходимо сделать
                                           False, чтобы координаты не изменились.
        :return: Дополненные source_data до всеобщего начала.
        """
        min_coordinate = np.min([self.coordinates[fileIndex][0] for fileIndex in range(len(self.files))])
        max_coordinate = np.max([self.coordinates[fileIndex][-1] for fileIndex in range(len(self.files))])

        for fileIndex in range(len(self.files)):
            # Удалим данные с файлов, которые записаны меньше чем на половине всего пути
            lower_bound = self.coordinates[fileIndex][0]    # Минимальная координата в self.coordinates[fileIndex]
            if lower_bound > (max_coordinate / 2):
                print(f"Файл {self.files[fileIndex]} является нерепрезентативным, поэтому он не будет анализироваться")
                self.files.pop(fileIndex)
                source_data.pop(fileIndex)
                self.coordinates.pop(fileIndex)
                self.marked_coordinates.pop(fileIndex)

        data = []    # Создадим список, в который скопируем source_data, чтобы в ходе выполнения source_data не изменился
        for fileIndex in range(len(self.files)):
            data.append(deepcopy(source_data[fileIndex]))

        middle_coordinate = ((max_coordinate / 2) // self.step) * self.step     # Выберем середину так, чтобы middle_coordinate была "кратна" self.step (тк мы работаем с
                                                                                # float, то остаток будет порядка 1e-14, поэтому говорить о "кратности" не совсем корректно).
                                                                                # Поэтому будем использовать np.isclose.
        lengths = []     # Список длин элементов data после обрезки
        for fileIndex in range(len(data)):
            data[fileIndex] = data[fileIndex][:np.where(np.isclose(self.coordinates[fileIndex], middle_coordinate))[0][0]]
            # Обрежем data[fileIndex], до того индекса, при котором self.coordinates[fileIndex] ближе всего к middle_coordinate
            # ([0][0] необходимо из-за особенности объекта, возвращаемого np.where())
            lengths.append(len(data[fileIndex]))

        sorted_lengths = np.unique(np.sort(lengths))        # Создадим отсортированный список длин элементов data
        # Продолжая пример из описания метода (без учёта обрезания), sorted_lengths = [185, 190, 200]

        reference_arraysIndex = []                          # Список индексов опорных массивов на различных участках заполнения

        for index in range(len(sorted_lengths)):            # Начинаем не с первого элемента, тк массивы длиной sorted_lengths[0] не будут опорными
            reference_arrayIndex = []                       # Список индексов опорных массивов на определённом участке заполнения
            for dataIndex in range(len(data)):
                if len(data[dataIndex]) >= sorted_lengths[index]:
                    reference_arrayIndex.append(dataIndex)

            reference_arraysIndex.append(reference_arrayIndex)

        # print('\n')
        # for dataIndex in range(len(data)):
        #     print(f'len(data[dataIndex]) = {len(data[dataIndex])}')
        # print()
        # print(f'sorted_lengths = {sorted_lengths}')

        # Если работаем с вертикальным профилем, то необходимо поднять графики, тк тангаж интегрируем, а курс нет.
        # Соответственно, вертикальный профиль начинается с нуля, а курс нет.
        if vertical_profileFlag:
            for lengthIndex in range(len(sorted_lengths) - 2, -1, -1):      # Пройдёмся по всем элементам sorted_lengths с конца кроме последнего
                # Комментарии для отладки в случае сбоя
                # print()
                # print('------------------------------------')
                # print(f'sorted_lengths[lengthIndex] = {sorted_lengths[lengthIndex]}')
                for dataIndex in range(len(data)):
                    if len(data[dataIndex]) == sorted_lengths[lengthIndex]:
                        # Мы проходимся по sorted_lengths с конца, чтобы сдвинуть массив предпоследний по длине, опираясь только на самый длинный массив.
                        # А двигать третий с конца по длине массив, опираясь только на последний и предпоследний по длине.

                        supportive_arraysIndex = []                         # Список индексов массивов с длиной больше sorted_lengths[lengthIndex]
                        supportive_length = []                              # Список длин массивов с длиной больше sorted_lengths[lengthIndex]
                        for supportive_dataIndex in range(len(data)):
                            if len(data[supportive_dataIndex]) > sorted_lengths[lengthIndex]:
                                supportive_arraysIndex.append(supportive_dataIndex)
                                supportive_length.append(len(data[supportive_dataIndex]))

                        # print()
                        # print(f'supportive_arraysIndex = {supportive_arraysIndex}')
                        # print(f'supportive_length = {supportive_length}')
                        # print(f'dataIndex = {dataIndex}')

                        source_data[dataIndex] += np.mean([source_data[index][supportive_length[supportive_arraysIndex.index(index)]
                                                                              - len(data[dataIndex])] for index in supportive_arraysIndex])
                        # Используем supportive_arraysIndex.index(index), чтобы найти длину массива в соответствии с его индексом в data,
                        # а не порядковым индексом в supportive_arraysIndex

            # Заново скопируем source_data в data и обрежем
            data = []
            for fileIndex in range(len(self.files)):
                data.append(deepcopy(source_data[fileIndex]))

            for fileIndex in range(len(data)):
                data[fileIndex] = data[fileIndex][:np.where(np.isclose(self.coordinates[fileIndex], middle_coordinate))[0][0]]

        for index in range(len(reference_arraysIndex)):
            for dataIndex in range(len(data)):
                if dataIndex not in reference_arraysIndex[index]:
                    data[dataIndex] = self.adding_data_from_beginning(data[dataIndex], [data[referenceIndex] for referenceIndex in reference_arraysIndex[index]])

        for index in range(len(data)):
            source_data[index] = np.append(data[index], source_data[index][np.where(np.isclose(self.coordinates[index], middle_coordinate))[0][0]:])

            if coordinate_changes_allowed:
                if index not in reference_arraysIndex[-1]:
                    self.coordinates[index] = np.append(np.arange(min_coordinate, self.coordinates[index][0], self.step), self.coordinates[index])

        return source_data

    @staticmethod
    def adding_data_from_beginning(array, reference_arrays):
        """
        Добавление данных к array в начало до максимальной точки отсчёта из всех данных в reference_arrays.
        :param array: массив, к которому добавляются данные.
        :param reference_arrays: список массивов на данные из которых основывается заполнение array.
        :return: дополненный array.
        """
        array_startIndex = [len(reference_arrays[index]) - len(array) - 1 for index in range(len(reference_arrays))]
        min_index = np.min(array_startIndex)
        while min_index >= 0:
            array = np.append(np.mean([reference_arrays[index][array_startIndex[index] - (np.min(array_startIndex) - min_index)]
                                       for index in range(len(reference_arrays))]), array)
            min_index -= 1

        return array

    def filling_ending(self, source_data, coordinate_changes_allowed=True):
        """
        Заполнение source_data до максимальной координаты, если файл записан не полностью.
        Например, есть три файла, записанные с 0 до 175, 200 и 160 метров соответственно.
        Тогда по выполнению данной функции все три файла будут заполнены до 200 метров таким образом:
        1. Третий файл продолжается до 175 метров средним значением с первого и второго файла без скачка после 160 метров.
        2. Первый и третий продолжаются значениями второго без скачка после 175 метров.

        Аналогично с любым количеством файлов.

        Важно!  Если данные в файле записаны до 100 метров (меньше половины максимальной длины), то такой файл считается нерепрезентативны и удаляется.

        :param source_data: Исходные данные.
        :param coordinate_changes_allowed: Тк может понадобиться дополнить сразу несколько величин, то этот флаг необходимо сделать
                                           False, чтобы координаты не изменились для повторного использования.
                                           Важно, чтобы в последнем изменении данных флаг был True, потому что self.coordinates[index] и
                                           source_data в конце должны оказаться одной размерности.
        :return: Дополненные source_data до всеобщего конца.
        """
        max_coordinate = np.max([self.coordinates[fileIndex][-1] for fileIndex in range(len(self.files))])

        for fileIndex in range(len(self.files)):
            # Удалим данные с файлов, которые записаны меньше чем на половине всего пути
            upper_bound = self.coordinates[fileIndex][-1]    # Максимальная координата в self.coordinates[fileIndex]
            if upper_bound < (max_coordinate / 2):
                print(f"Файл {self.files[fileIndex]} является нерепрезентативным, поэтому он не будет анализироваться")
                self.files.pop(fileIndex)
                source_data.pop(fileIndex)
                self.coordinates.pop(fileIndex)
                self.marked_coordinates.pop(fileIndex)

        data = []    # Создадим список, в который скопируем source_data, чтобы в ходе выполнения source_data не изменился
        for fileIndex in range(len(self.files)):
            data.append(deepcopy(source_data[fileIndex]))

        middle_coordinate = ((max_coordinate / 2) // self.step) * self.step     # Выберем середину так, чтобы middle_coordinate была "кратна" self.step (тк мы работаем с
                                                                                # float, то остаток будет порядка 1e-14, поэтому говорить о "кратности" не совсем корректно).
        lengths = []     # Список длин элементов data после обрезки
        for fileIndex in range(len(data)):
            data[fileIndex] = data[fileIndex][np.where(np.isclose(self.coordinates[fileIndex], middle_coordinate))[0][0]:]
            # Обрежем data[fileIndex], начиная с того индекса, при котором self.coordinates[fileIndex] ближе всего к middle_coordinate
            # ([0][0] необходимо из-за особенности объекта, возвращаемого np.where())
            lengths.append(len(data[fileIndex]))

        sorted_lengths = np.unique(np.sort(lengths))        # Создадим отсортированный список длин элементов data
        # Продолжая пример из описания метода (без учёта обрезания), sorted_lengths = [160, 175, 200]

        reference_arraysIndex = []                          # Список индексов опорных массивов на различных участках заполнения

        for index in range(1, len(sorted_lengths)):         # Начинаем не с первого элемента, тк массивы длиной sorted_lengths[0] не будут опорными
            reference_arrayIndex = []                       # Список индексов опорных массивов на определённом участке заполнения
            for dataIndex in range(len(data)):
                if len(data[dataIndex]) >= sorted_lengths[index]:
                    reference_arrayIndex.append(dataIndex)

            reference_arraysIndex.append(reference_arrayIndex)

        # Из примера (без учёта обрезания): reference_arraysIndex[0] = [0, 1]
        #                                   // соответствует участку 160 - 175 метров
        #                                   // 0 - индекс первого массива
        #                                   // 1 - индекс второго массива
        #
        #                                   reference_arraysIndex[1] = [1]
        #                                   // соответствует участку 175 - 200 метров

        for index in range(len(reference_arraysIndex)):
            for dataIndex in range(len(data)):
                if dataIndex not in reference_arraysIndex[index]:
                    data[dataIndex] = self.adding_data_from_end(data[dataIndex], [data[referenceIndex] for referenceIndex in reference_arraysIndex[index]])

        for index in range(len(data)):
            source_data[index] = np.append(source_data[index][:np.where(np.isclose(self.coordinates[index], middle_coordinate))[0][0]], data[index])
            if coordinate_changes_allowed:
                if index not in reference_arraysIndex[-1]:
                    self.coordinates[index] = np.append(self.coordinates[index][:-1], np.arange(self.coordinates[index][-1], max_coordinate, self.step))

        return source_data

    @staticmethod
    def adding_data_from_end(array, reference_arrays):
        """
        Добавление в конец array средних значений reference_arrays в каждой точке.

        :param array: массив, к которому будут добавлены данные.
        :param reference_arrays: массивы, данные из которых будут использованы при добавлении к array.
        :return: дополненный array
        """
        array_startIndex = [len(reference_arrays[index]) - len(array) for index in range(len(reference_arrays))]
        min_index = np.min(array_startIndex)
        # ordIndex (ordinal index) - порядковый индекс
        cur_index = 0
        length = len(array)
        shift = np.mean([reference_arrays[ordIndex][length] for ordIndex in range(len(reference_arrays))]) - array[-1]

        while cur_index < min_index:
            array = np.append(array, np.mean([reference_arrays[index][length + cur_index] for index in range(len(reference_arrays))]) - shift)
            cur_index += 1

        return array

    @staticmethod
    def approximate(x1, y1, x2, y2):
        """
        Решение системы
        | y1 = k * x1 + b
        | y2 = k * x2 + b

        :return: [k, b]
        """
        return np.linalg.inv(np.array([[x1, 1], [x2, 1]])).dot(np.array([y1, y2]))

    def line_coefficients(self, x_coordinate, x_axis, y_axis):
        """
        Получение коэффициентов линии, проходящей при х = x_coordinate.
        :param x_coordinate: Заданная координата.
        :param x_axis: Массив координат.
        :param y_axis: Массив значений данных.
        :return: [k, b]
        """
        min_index = np.argmin(np.abs(x_coordinate - x_axis))    # Индекс ближайшего элемента x_axis к x_coordinate

        # Затем, в зависимости от того, где находится x_coordinate относительно x_axis[min_index] (справа или слева),
        # вызывается self.approximate с соответствующими точками.

        if x_coordinate > x_axis[min_index]:
            line = self.approximate(x_axis[min_index - 1], y_axis[min_index - 1], x_axis[min_index], y_axis[min_index])
        else:
            line = self.approximate(x_axis[min_index], y_axis[min_index], x_axis[min_index + 1], y_axis[min_index + 1])

        return line

    ############# Получение общих отмеченных точек #############

    def get_common_marked_coordinates(self):

        # Ограничим self.marked_coordinates верхней и нижней границами self.coordinates
        for index in range(len(self.marked_coordinates)):
            self.marked_coordinates[index] = self.marked_coordinates[index][np.where((self.marked_coordinates[index] > self.coordinates[0]) &
                                                                                     (self.marked_coordinates[index] < self.coordinates[-1]))]
            # И развернём массив если его первый элемент больше последнего
            if self.marked_coordinates[index][0] > self.marked_coordinates[index][-1]:
                self.marked_coordinates[index] = np.flip(self.marked_coordinates[index])

        # Создадим массив для общих отмеченных координат длины n, где n - результат применения медианного фильтра к длинам self.marked_coordinates[index]
        common_marked_coordinates = np.zeros(self.median_filter([len(self.marked_coordinates[index]) for index in range(len(self.marked_coordinates))]))

        # Затем возьмём массив из self.marked_coordinates с длиной n в качестве опорного
        index = -1
        length = -1
        while length != len(common_marked_coordinates):
            index += 1
            length = len(self.marked_coordinates[index])

        reference_array = self.marked_coordinates[index]    # Опорный массив

        # Сравним каждый элемент опорного массива с пятью элементами с близкими индексами из self.marked_coordinates
        for index in range(len(reference_array)):
            close_coordinates = []      # Список, куда будут сохраняться ближайшие элементы из self.marked_coordinates относительно reference_array[index]
            # Пройдёмся по пяти элементам с близким индексом
            for supportive_index in range(index - 2, index + 3):
                # Проверим, что supportive_index > 0, чтобы не проверять элементы с конца каждого массива
                if supportive_index >= 0:
                    # Теперь пройдёмся по всем массивам из self.marked_coordinates
                    for dataIndex in range(len(self.marked_coordinates)):
                        # Т.к. массивы в self.marked_coordinates не обязаны быть одинаковой длины, то можно выйти за пределы какого-то из массивов
                        try:
                            # Критерий близости - разница между координатами меньше 5 метров
                            if abs(reference_array[index] - self.marked_coordinates[dataIndex][supportive_index]) < 5:
                                close_coordinates.append(self.marked_coordinates[dataIndex][supportive_index])
                        except IndexError:
                            pass

            common_marked_coordinates[index] = np.mean(close_coordinates)

        self.marked_coordinates = common_marked_coordinates
        # self.marked_coordinates = self.fake_get_common_marked_coordinates(10)

    def fake_get_common_marked_coordinates(self, n):
        """
        Создание n фиктивных точек стяжки.
        :param n: Количество точек.
        :return: Массив фиктивных точек
        """
        return np.linspace(self.coordinates[0] + self.step, self.coordinates[-1] - self.step, n)

    ############# Построение графиков ###############

    def charting(self, x_axis, y_axis, x_axis1D=True, title=None, label=None, linewidth: int = 2, x_label='Пройденный путь, м', y_label=None,
                 x_points=None, y_points=None, x_lim=None, y_lim=None, annotation=None, saved_dir=None, saved_name=None):

        """
        Построение графиков по величинам x_axis и y_axis

        :param x_axis: Данные для оси Х.
        :param y_axis: Данные для оси У, представленные в виде многомерного списка/массива.
                       Поэтому одномерные данные необходимо передавать в виде кортежа или списка.
                       Пример: (array, ).
        :param x_axis1D: Флаг, означающий многомерность или одномерность x_axis.
                         True ---> x_axis - одномерный.
                         False ---> x_axis - многомерный.
                         ВАЖНО!!! Если x_axis многомерный, то его размерность должна совпадать с размерностью y_axis.
        :param title: Название графика.
        :param label: Многомерный список подписей, которые будут добавлены на график к каждому элементу y_axis.
                      Поэтому если y_axis - одномерный список/массив, то label необходимо передавать так же как и y_axis
                      в виде кортежа или списка.
                      Если label не указан, то подписи не будут нанесены на график.
        :param linewidth: Толщина линий на графике.
        :param x_label: Подпись к оси абсцисс.
        :param y_label: Подпись к оси ординат.
        :param x_points: Дополнительные абсциссы точек, которые необходимо нанести на график с помощью scatter.
        :param y_points: Дополнительные ординаты точек, которые необходимо нанести на график с помощью scatter.
        :param x_lim: Пределы по оси абсцисс.
        :param y_lim: Пределы по оси ординат.
        :param annotation: Дополнительный текст, который будет добавлен на график в рамке.
        :param saved_dir: Имя дополнительной папки в self.dir/Стяжка, куда будет сохранён график.
        :param saved_name: Имя, с которым график будет сохранён в папке self.dir/Стяжка/saved_dir/Графики. Если оно не указано, то график сохранён не будет.
        """
        fig, ax = plt.subplots(tight_layout=True)

        fig.set_figheight(9)
        fig.set_figwidth(16)

        if x_axis1D:
            for index in range(len(y_axis)):
                if label:
                    ax.plot(x_axis, y_axis[index] * 1000, label=label[index], linewidth=linewidth)
                    ax.legend()
                else:
                    ax.plot(x_axis, y_axis[index] * 1000, linewidth=linewidth)

        else:
            for index in range(len(y_axis)):
                if label:
                    ax.plot(x_axis[index], y_axis[index] * 1000, label=label[index], linewidth=linewidth)
                    ax.legend()
                else:
                    ax.plot(x_axis[index], y_axis[index] * 1000, linewidth=linewidth)

        ax.grid()

        ax.set_title(title, weight='bold', fontsize=16)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if x_points and y_points:
            for index in range(len(x_points)):
                ax.scatter(x_points[index], y_points[index], zorder=10, edgecolor="black")

        if x_lim:
            ax.set_xlim(x_lim)

        if y_lim:
            ax.set_ylim(y_lim)

        if annotation:
            ax.annotate(annotation, xy=(0.6, 0.9), xycoords='axes fraction', size=14,
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", ec="gray", lw=2))

        if saved_name:
            if not os.path.exists(self.dir + "/Стяжка"):
                os.mkdir(self.dir + "/Стяжка")

            if not os.path.exists(self.dir + f"/Стяжка/{saved_dir}"):
                os.mkdir(self.dir + f"/Стяжка/{saved_dir}")

            if not os.path.exists(self.dir + f"/Стяжка/{saved_dir}/Графики"):
                os.mkdir(self.dir + f"/Стяжка/{saved_dir}/Графики")

            fig.savefig(f'{self.dir}/Стяжка/{saved_dir}/Графики/{saved_name}.png')

    ############# Интегрирование #############

    @staticmethod
    def integration(x_value, y_value):
        """
        Интегрирование с помощью метода трапеции.
        """

        result = 0
        integrated_array = np.zeros_like(x_value)

        for index in range(len(x_value) - 1):
            result += (x_value[index + 1] - x_value[index]) * (y_value[index] + y_value[index + 1]) / 2
            integrated_array[index + 1] = result

        return integrated_array

    ############# Стяжка #############

    def tightening(self, source_data, vertical_profileFlag=False):
        previous_index = 0

        # Пройдёмся по каждой точке, по которой будет происходить стяжка
        for markIndex in range(len(self.marked_coordinates)):

            # Найдём индекс ближайшего элемента self.coordinates к отмеченной точке. Критерий близости - разница между элементами меньше половины self.step
            index = np.where(np.isclose(self.coordinates, self.marked_coordinates[markIndex], atol=self.step / 2))[0][0]

            # Фильтрованная высота, полученная с помощью медианного или среднего фильтра (в зависимости от self.filter_type)
            filtered_height = self.filters[self.filter_type]([source_data[dataIndex][index] for dataIndex in range(len(source_data))])

            # Сведём точки source_data[dataIndex][index] в одну (как косичку в местах, где есть резинки)
            for dataIndex in range(len(source_data)):
                line_coefficients = (source_data[dataIndex][index] - filtered_height) / (self.coordinates[index] - self.coordinates[previous_index])
                for i in range(previous_index, len(self.coordinates)):
                    source_data[dataIndex][i] -= line_coefficients * (self.coordinates[i] - self.coordinates[previous_index])

            previous_index = index      # Каждый раз previous_index смещается для того, чтобы наклонять не весь график, а только начиная с previous_index

        # Сведём последние точки в self.end_height для вертикального профиля
        # А для других данных сведём графики в конце к среднему значению source_data[dataIndex][-1]
        if not vertical_profileFlag:
            self.end_height = np.mean([source_data[index][-1] for index in range(len(source_data))])

        for dataIndex in range(len(source_data)):
            line_coefficients = (source_data[dataIndex][-1] - self.end_height) / (self.coordinates[-1] - self.coordinates[previous_index])
            for i in range(previous_index, len(self.coordinates)):
                source_data[dataIndex][i] -= line_coefficients * (self.coordinates[i] - self.coordinates[previous_index])

        return source_data

    ############# Сохранение данных в CSV файл #############

    def writing_to_CSV_file(self, data, saved_dir, file_name):
        """
        Сохраним значения data и СКО в CSV файл ./self.dir/Стяжка/file_name
        """
        if not os.path.exists(self.dir + "/Стяжка"):
            os.mkdir(self.dir + "/Стяжка")

        if not os.path.exists(self.dir + f"/Стяжка/{saved_dir}"):
            os.mkdir(self.dir + f"/Стяжка/{saved_dir}")

        csv_file = open(f'{self.dir}/Стяжка/{saved_dir}/{file_name}.csv', 'w')

        # Напишем первую стоку
        csv_file.write('Путь')
        for file in self.files:
            csv_file.write(f' {file}')
        csv_file.write(' СКО\n')

        # Запишем все данные
        for index in range(len(self.coordinates)):
            csv_file.write(f'{self.to_csv_format(self.coordinates[index])}')

            for dataIndex in range(len(data)):
                csv_file.write(f' {self.to_csv_format(data[dataIndex][index])}')

            csv_file.write(f' {self.to_csv_format(self.std_array[index])}\n')

        csv_file.close()

    @staticmethod
    def to_csv_format(value):
        """
        Замена точки на запятую в числе для записи в CSV файл.
        :param value: Число с точкой.
        :return: Строка, содержащая исходное число с запятой с восемью знаками после запятой.
        """
        return str(round(value, 8)).replace(".", ",")

    ############# Фильтры #############

    @staticmethod
    def median_filter(arr):
        return np.sort(arr)[len(arr) // 2]

    @staticmethod
    def average_filter(arr):
        return np.mean(arr)

    #################################
