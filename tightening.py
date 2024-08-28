# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from copy import deepcopy
# from pprint import pprint


class measuring:
    def __init__(self, dir, titleIndex, filter_type, end_height):
        """
        Класс, предназначенный для "стягивания" графиков в определённых точка

        :param dir: Директория расположения CSV файлов, относительно расположения программы.
        :param titleIndex: Величина, по которой будет происходить "стяжка"
        :param filter_type: Используемый фильтр.
                            Возможные варианты: average_filter - фильтр усреднения
                                                median_filter - медианный фильтр
        :param end_height: Высота в конце проезда, относительно его начала.
        """

        self.dir = dir
        self.titleIndex = titleIndex
        self.filter_type = filter_type
        self.end_height = end_height

        if self.filter_type not in ('average_filter', 'median_filter'):
            print('Неправильно выбранный фильтр')
            exit(1)

        # Функции используемого фильтра
        self.filters = {'average_filter': self.average_filter,
                        'median_filter': self.median_filter}

        # Информация для построения графиков
        self.chart_info = {'average_filter': {'Заголовок': 'СКО, полученное усреднением',
                                              'Имя файла': 'СКО_усреднение'},
                           'median_filter': {'Заголовок': 'СКО, полученное медианным фильтром',
                                             'Имя файла': 'СКО_медианный фильтр'}}

        self.files = self.get_files()       # Список CSV файлов
        self.data = []                      # Список данных величины titleIndex, прочитанный из files
        self.coordinates = []               # Список массивов координат, прочитанный из files. После self.interpolation станет массивом координат
        self.marked_coordinates = []        # Список массивов координат, прочитанный из files, которые отмечены отметкой Piket.
                                            # После self.get_common_marked_coordinates он станет массивом отмеченных координат. По этим координатам будет происходить "стяжка"
        self.std_array = np.ndarray         # Массив среднеквадратичных отклонений

        self.step = 0.22553                 # Шаг с которым будет проводиться интерполяция

        self.start()

    def __del__(self):
        # plt.show()
        pass

    ##################################################

    def start(self):
        self.reading_values()
        self.interpolation()
        self.get_common_marked_coordinates()
        # pprint(self.marked_coordinates)

        # Проинтегрируем self.data
        for index in range(len(self.data)):
            self.data[index] = self.integration(self.coordinates, self.data[index])

        # self.charting(self.coordinates, self.data, label=self.files)

        # Сведём графики вертикального профиля в последней точке
        for index in range(len(self.data)):
            for i in range(len(self.data[index])):
                self.data[index][i] -= (self.data[index][-1] - self.end_height) / self.coordinates[-1] * (self.coordinates[i] - self.coordinates[0])

        self.charting(self.coordinates, self.data,
                      linewidth=2.5,
                      label=self.files,
                      title='Вертикальный профиль, сведённый в последней точке',
                      saved_name="Сырой профиль"
                      )

        self.tightening()

        # Заполним массив СКО
        self.std_array = np.array([np.std([self.data[dataIndex][index] for dataIndex in range(len(self.data))]) for index in range(len(self.coordinates))])

        self.charting(self.coordinates,
                      (self.std_array, ),
                      linewidth=3,
                      title=self.chart_info[self.filter_type]['Заголовок'],
                      saved_name=self.chart_info[self.filter_type]['Имя файла'],
                      annotation=f'Количество файлов - {len(self.files)}\nКоличество точек сведения - {len(self.marked_coordinates)}'
                      )

        self.writing_to_CSV_file()

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

        return files

    ############# Чтение данных #############

    def reading_values(self):
        """
        Чтение данных с последующим сохранением их в self.data, self.coordinates, self.marked_coordinates
        """

        for fileIndex in range(len(self.files)):

            try:
                self.data.append(self.clearing(np.loadtxt(f'{self.dir}/{self.files[fileIndex]}', dtype=str, skiprows=1, unpack=True, encoding="UTF-8")[self.titleIndex]) * np.pi / 180)
                self.coordinates.append(self.clearing(np.loadtxt(f'{self.dir}/{self.files[fileIndex]}', dtype=str, skiprows=1, unpack=True, encoding="UTF-8")[0]))

                marks = np.loadtxt(f'{self.dir}/{self.files[fileIndex]}', dtype=str, skiprows=1, unpack=True, encoding="UTF-8")[3]
                marked_coordinates = np.zeros(0)
                for markIndex in range(len(marks)):
                    if marks[markIndex] == 'Piket':
                        marked_coordinates = np.append(marked_coordinates, self.coordinates[fileIndex][markIndex])

                self.marked_coordinates.append(marked_coordinates)

            except UnicodeDecodeError:      # Ошибка, возникающая, если в self.dir присутствуют другие CSV файлы, который не подходят под шаблон CSV файла с измерениями
                print(f'Ошибка чтения {self.files[fileIndex]}')

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

    def interpolation(self):
        """
        Получение значений self.data в одних и тех же координатах с шагом self.step с диапазоном от наибольшей минимальной координаты до наименьшей максимальной координаты.
        После выполнения функции список массивов координат self.coordinates станет массивом координат с шагом self.step с таким же диапазоном.
        """

        for fileIndex in range(len(self.files)):
            min_coordinate = self.coordinates[fileIndex][0] if 'out' in self.files[fileIndex] else self.coordinates[-1]
            max_coordinate = self.coordinates[fileIndex][-1] if 'out' in self.files[fileIndex] else self.coordinates[0]
            # Тк в файлах *_out.csv координаты идут по возрастанию, а в *_in.csv по убыванию

            coord = (min_coordinate // self.step + 1) * self.step       # Таким образом мы гарантируем, что coord будет кратен self.step и больше
                                                                        # min_coordinate, что необходимо для корректной работы self.line_coefficients,
                                                                        # тк coord должна всегда находится между двумя точками из self.coordinates[fileIndex]
            coord_array = np.zeros(1)
            file_data_array = np.zeros(1)

            line = self.line_coefficients(coord, self.coordinates[fileIndex], self.data[fileIndex])

            coord_array[0] = coord
            file_data_array[0] = line[0] * coord + line[1]

            coord += self.step

            while coord < max_coordinate:
                line = self.line_coefficients(coord, self.coordinates[fileIndex], self.data[fileIndex])
                coord_array = np.append(coord_array, coord)
                file_data_array = np.append(file_data_array, line[0] * coord + line[1])
                coord += self.step

            self.coordinates[fileIndex] = coord_array
            self.data[fileIndex] = file_data_array

        self.filling_beginning()
        self.filling_ending()

    def filling_beginning(self):
        pass

    def filling_ending(self):
        """
        Заполнение данных до максимальной координаты, если файл записан не полностью.
        Например, есть три файла, записанные с 0 до 175, 200 и 160 метров соответственно.
        Тогда по выполнению данной функции все три файла будут заполнены до 200 метров таким образом:
        1. Третий файл продолжается до 175 метров средним значением с первого и второго файла без скачка после 160 метров.
        2. Первый и третий продолжаются значениями второго без скачка после 175 метров.

        Аналогично с любым количеством файлов.

        Важно!  Если данные в файле записаны до 95 метров (меньше половины максимальной длины), то такой файл считается нерепрезентативны и удаляется.
        """
        max_coordinate = np.max([self.coordinates[fileIndex][-1] if 'out' in self.files[fileIndex] else self.coordinates[0] for fileIndex in range(len(self.files))])
        # Тк в файлах *_out.csv координаты идут по возрастанию, а в *_in.csv по убыванию

        for fileIndex in range(len(self.files)):
            # Удалим данные с файлов, которые записаны меньше чем на половине всего пути
            upper_bound = self.coordinates[fileIndex][0] if 'out' in self.files[fileIndex] else self.coordinates[-1]    # Максимальная координата в self.coordinates[fileIndex]
            if upper_bound < (max_coordinate / 2):
                print(f"Файл {self.files[fileIndex]} является нерепрезентативным, поэтому он не будет анализироваться")
                self.files.pop(fileIndex)
                self.data.pop(fileIndex)
                self.coordinates.pop(fileIndex)
                self.marked_coordinates.pop(fileIndex)

        data = []    # Создадим список, в который скопируем self.data, чтобы в ходе выполнения self.data не изменился
        for fileIndex in range(len(self.files)):
            data.append(deepcopy(self.data[fileIndex]))

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
        length = len(self.marked_coordinates[0])
        index = 0
        while length != len(common_marked_coordinates):
            length += 1
            index += 1

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

    def charting(self, x_axis, y_axis, title=None, label=None, linewidth: int = 2, saved_name=None, annotation=None):
        """
        Построение графиков по величинам x_axis и y_axis

        :param x_axis: Данные для оси Х, представленные в виде одномерного списка/массива.
        :param y_axis: Данные для оси У, представленные в виде многомерного списка/массива.
                       Поэтому одномерные данные необходимо передавать в виде кортежа или списка.
                       Пример: (array, ).
        :param title: Название графика.
        :param label: Многомерный список подписей, которые будут добавлены на график к каждому элементу y_axis.
                      Поэтому если y_axis - одномерный список/массив, то label необходимо передавать так же как и y_axis
                      в виде кортежа или списка.
                      Если label не указан, то подписи не будут нанесены на график.
        :param linewidth: Толщина линий на графике.
        :param saved_name: Имя, с которым график будет сохранён в папке self.dir/Стягивание/Графики. Если оно не указано, то график сохранён не будет.
        :param annotation: Дополнительный текст, который будет добавлен на график в рамке.
        """

        fig, ax = plt.subplots(tight_layout=True)

        fig.set_figheight(9)
        fig.set_figwidth(16)

        for index in range(len(y_axis)):
            if label:
                ax.plot(x_axis, y_axis[index] * 1000, label=label[index], linewidth=linewidth)
                ax.legend()
            else:
                ax.plot(x_axis, y_axis[index] * 1000, linewidth=linewidth)

        ax.grid()

        ax.set_title(title, weight='bold', fontsize=16)
        ax.set_xlabel('Пройденный путь, м')
        ax.set_ylabel('Перепад высот, мм')

        if annotation:
            ax.annotate(annotation, xy=(0.6, 0.9), xycoords='axes fraction', size=14,
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", ec="gray", lw=2))

        if saved_name:
            if not os.path.exists(self.dir + "/Стяжка"):
                os.mkdir(self.dir + "/Стяжка")

            if not os.path.exists(self.dir + "/Стяжка/Графики"):
                os.mkdir(self.dir + "/Стяжка/Графики")

            fig.savefig(f'{self.dir}/Стяжка/Графики/{saved_name}.png')

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

    def tightening(self):
        previous_index = 0

        # Пройдёмся по каждой точке, по которой будет происходить стяжка
        for markIndex in range(len(self.marked_coordinates)):

            # Найдём индекс ближайшего элемента self.coordinates к отмеченной точке. Критерий близости - разница между элементами меньше половины self.step
            index = np.where(np.isclose(self.coordinates, self.marked_coordinates[markIndex], atol=self.step / 2))[0][0]

            # Фильтрованная высота, полученная с помощью медианного или среднего фильтра (в зависимости от self.filter_type)
            filtered_height = self.filters[self.filter_type]([self.data[dataIndex][index] for dataIndex in range(len(self.data))])

            # Сведём точки self.data[dataIndex][index] в одну (как косичку в местах, где есть резинки)
            for dataIndex in range(len(self.data)):
                line_coefficients = (self.data[dataIndex][index] - filtered_height) / (self.coordinates[index] - self.coordinates[previous_index])
                for i in range(previous_index, len(self.coordinates)):
                    self.data[dataIndex][i] -= line_coefficients * (self.coordinates[i] - self.coordinates[previous_index])

            previous_index = index      # Каждый раз previous_index смещается для того, чтобы наклонять не весь график, а только начиная с previous_index

        # Сведём последние точки в self.end_height
        for dataIndex in range(len(self.data)):
            line_coefficients = (self.data[dataIndex][-1] - self.end_height) / (self.coordinates[-1] - self.coordinates[previous_index])
            for i in range(previous_index, len(self.coordinates)):
                self.data[dataIndex][i] -= line_coefficients * (self.coordinates[i] - self.coordinates[previous_index])

        self.charting(self.coordinates, self.data,
                      label=self.files,
                      title='Сведённый вертикальный профиль',
                      saved_name="Сведённый профиль")

    ############# Сохранение данных в CSV файл #############

    def writing_to_CSV_file(self):
        """
        Сохраним значения сведённого профиля и СКО в CSV файл ./self.dir/Стяжка/file_name
        """
        if not os.path.exists(self.dir + "/Стяжка"):
            os.mkdir(self.dir + "/Стяжка")

        file_name = 'Сведённый профиль_усреднение.csv' if self.filter_type == 'average_filter' else 'Сведённый профиль_медианный фильтр.csv'

        csv_file = open(f'{self.dir}/Стяжка/{file_name}', 'w')

        # Напишем первую стоку
        csv_file.write('Путь')
        for file in self.files:
            csv_file.write(f' {file}')
        csv_file.write(' СКО\n')

        # Запишем все данные
        for index in range(len(self.coordinates)):
            csv_file.write(f'{self.to_csv_format(self.coordinates[index])}')

            for dataIndex in range(len(self.data)):
                csv_file.write(f' {self.to_csv_format(self.data[dataIndex][index])}')

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


# parser = argparse.ArgumentParser()                                                          # Создадим парсер для анализа аргументов из командной строки
# parser.add_argument("-eh", "--end_height", type=float, default=0,
#                     help="Разница высоты между первой и последней точками измерения")       # Добавим в него необязательный аргумент end_height, который передадим
#                                                                                             # в конструктор класса measuring в качестве параметра end_height
#                                                                                             # Значение по умолчанию - 0
#
# self_dir = './'                                                                             # Директория в которой будет работать стягивание данных
# measuring(self_dir, 7, 'average_filter', parser.parse_args().end_height)                    # 7 - номер столбца с Креном

self_dir = './CSV files/17.07.24'                                                                             # Директория в которой будет работать стягивание данных
measuring(self_dir, 7, 'average_filter', 1.2)                    # 7 - номер столбца с Креном

