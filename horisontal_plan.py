# -*- coding: utf-8 -*-

from measuring import *
from progress.bar import ChargingBar


class HorizontalPlan(Measuring):
    def __init__(self, dir, filter_type, end_height, testFlag):
        """
        Класс для анализа горизонтального плана правого и левого рельса по CSV файлам из директории dir.

        :param dir: Директория расположения сохранённых файлов, относительно расположения программы.
        :param filter_type: Используемый фильтр.
                            Возможные варианты: average_filter - фильтр усреднения,
                                                median_filter - медианный фильтр.
        :param end_height: Высота в конце проезда, относительно его начала.
        :param testFlag: Флаг тестового режима. Если True, то по ходу выполнения программы
                         будут появляться вспомогательные изображения, поясняющие выполняемые шаги.
        """

        super().__init__(dir, filter_type, end_height, testFlag=testFlag)

        self.course = []        # Значение курса телеги
        self.saved_dir = 'Горизонтальный план'

        # Информация для построения графиков
        self.chart_info = {'average_filter': {'Заголовок': 'СКО курса, полученное усреднением',
                                                    'Имя файла': 'СКО курса_усреднение'},
                           'median_filter': {'Заголовок': 'СКО курса, полученное медианным фильтром',
                                             'Имя файла': 'СКО курса_медианный фильтр'}
                           }

        self.file_namesInfo = {'average_filter': 'Сведённый курс_усреднение',
                               'median_filter': 'Сведённый курс_медианный фильтр'
                               }

        self.start()

    def start(self):
        print('Построение курса')
        bar = ChargingBar('Выполнение программы: ', max=8)    # Шкала выполнения

        self.course = self.reading_values()[0]
        bar.next()
        bar.next()

        self.course = self.interpolation(course=self.course)[0]
        bar.next()

        if self.testFlag:
            self.charting(self.coordinates, self.course, x_axis1D=False, title=f'Изначальные данные курса',
                          label=self.files, saved_dir=self.saved_dir,
                          saved_name='Изначальные данные')

        self.course = self.filling_beginning(self.course)
        bar.next()

        self.course = self.filling_ending(self.course)
        bar.next()

        for dataIndex in range(len(self.course)):
            # Из-за особенности работы компьютера с числами с плавающей точкой может возникнуть ситуация, когда в self.coordinates[dataIndex]
            # сохранены почти одинаковые координаты (разница между ними порядка 1е-8).
            # Поэтому удостоверимся, что такой ситуации нет.
            # А если есть, тогда удалим повторяющуюся координату и данные, соответсвующее ей.
            try:
                double_infoIndex = np.where(np.isclose(np.diff(self.coordinates[dataIndex]), 0))[0][0]
                self.course[dataIndex] = np.delete(self.course[dataIndex], double_infoIndex)
                self.coordinates[dataIndex] = np.delete(self.coordinates[dataIndex], double_infoIndex)
            except IndexError:
                pass

        if self.testFlag:
            self.charting(self.coordinates, self.course, x_axis1D=False,
                          title=f'Продолженные данные курса', linewidth=2.5,
                          saved_dir=self.saved_dir, saved_name='Продолженные данные')

        # Тк у нас все self.coordinates[index] одинаковы, то нет смысла хранить их как список, поэтому сделаем приведение типа.
        self.coordinates = self.coordinates[0]

        self.charting(self.coordinates, self.course,
                      linewidth=2.5,
                      label=self.files,
                      title=f'Исходные данные для курса',
                      saved_dir=self.saved_dir,
                      saved_name=f'Сырой курс'
                      )

        self.get_common_marked_coordinates()
        bar.next()

        self.course = self.tightening(self.course)

        self.charting(self.coordinates, self.course,
                      label=self.files,
                      title=f'Сведённый курс',
                      saved_dir=self.saved_dir,
                      saved_name=f'Сведённый курс')

        bar.next()

        # Заполним массив СКО
        self.std_array = np.array([np.std([self.course[dataIndex][index] for dataIndex in range(len(self.course))]) for index in range(len(self.coordinates))])

        self.charting(self.coordinates,
                      (self.std_array, ),
                      linewidth=3,
                      title=self.chart_info[self.filter_type]['Заголовок'],
                      saved_dir=self.saved_dir,
                      saved_name=self.chart_info[self.filter_type]['Имя файла'],
                      annotation=f'Количество файлов - {len(self.files)}\nКоличество точек сведения - {len(self.marked_coordinates)}'
                      )

        self.writing_to_CSV_file(self.course, self.saved_dir, self.file_namesInfo[self.filter_type])
        bar.next()

        print('\nУспешное завершение программы\n')
