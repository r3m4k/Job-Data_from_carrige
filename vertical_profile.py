# -*- coding: utf-8 -*-

from measuring import *
from progress.bar import ChargingBar


class VerticalProfile(Measuring):
    def __init__(self, dir, filter_type, end_height, testFlag, rotationFlag):
        """
        Класс для анализа вертикального профиля правого и левого рельса по CSV файлам из директории dir.

        :param dir: Директория расположения сохранённых файлов, относительно расположения программы.
        :param filter_type: Используемый фильтр.
                            Возможные варианты: average_filter - фильтр усреднения,
                                                median_filter - медианный фильтр.
        :param end_height: Высота в конце проезда, относительно его начала.
        :param testFlag: Флаг тестового режима. Если True, то по ходу выполнения программы
                         будут появляться вспомогательные изображения, поясняющие выполняемые шаги.
        :param rotationFlag: Флаг поворота гироскопа в БИНСе относительно направления движения тележки.
                             True - гироскоп расположен перпендикулярно направлению пути.
                             False - гироскоп расположен вдоль направления движения.
        """

        super().__init__(dir, filter_type, end_height, testFlag=testFlag)

        self.rotationFlag = rotationFlag

        self.roll = []                      # Фактический крен телеги
        self.pitch = []                     # Фактический тангаж телеги
        self.vertical_profile = []          # Будущий вертикальный профиль
        self.track_width = []               # Ширина колеи

        self.saved_dir = 'Вертикальный профиль'

        # Информация для построения графиков
        self.chart_info = {'average_filter': {'Заголовок': 'СКО вертикального профиля (мм), полученное усреднением',
                                                           'Имя файла': 'СКО вертикального профиля_усреднение'},
                           'median_filter': {'Заголовок': 'СКО вертикального профиля (мм), полученное медианным фильтром',
                                             'Имя файла': 'СКО вертикального профиля_медианный фильтр'}
                           }

        self.file_namesInfo = {'average_filter': 'Сведённый вертикальный профиль_усреднение',
                               'median_filter': 'Сведённый вертикальный профиль_медианный фильтр'
                               }

        self.start()

    def start(self):
        print('Построение вертикального профиля')

        bar = ChargingBar('Выполнение программы: ', max=10)    # Шкала выполнения

        if self.rotationFlag:
            _, self.pitch, self.roll, self.track_width = self.reading_values()
        else:
            _, self.roll, self.pitch, self.track_width = self.reading_values()

        bar.next()

        _, self.roll, self.pitch, self.track_width = self.interpolation(roll=self.roll, pitch=self.pitch, track_width=self.track_width)
        bar.next()

        # Проинтегрируем тангаж, чтобы получить вертикальный профиль
        for index in range(len(self.pitch)):
            self.vertical_profile.append(self.integration(self.coordinates[index], self.pitch[index]))

        bar.next()

        if self.testFlag:
            self.charting(self.coordinates, self.pitch, x_axis1D=False, title='Фактический тангаж телеги',
                          label=self.files, linewidth=2, y_label='радианы * 0,001',
                          saved_dir=self.saved_dir, saved_name='Тангаж телеги')

            self.charting(self.coordinates, self.track_width, x_axis1D=False, label=self.files,
                          title='Изначальная ширина колеи (мм)', saved_dir=self.saved_dir, saved_name='Ширина колеи в начале')

        # Сведём графики вертикального профиля в последней точке
        self.fitting_to_finalHeight(self.vertical_profile)

        if self.testFlag:
            self.charting(self.coordinates, self.vertical_profile, x_axis1D=False, label=self.files, y_label='Перепад высот, мм',
                          title='Изначальные данные вертикального профиля', saved_dir=self.saved_dir, saved_name='Изначальные данные')

        bar.next()

        self.vertical_profile = self.filling_beginning(self.vertical_profile, vertical_profileFlag=True, coordinate_changes_allowed=False)
        self.roll = self.filling_beginning(self.roll, coordinate_changes_allowed=False)
        self.track_width = self.filling_beginning(self.track_width)

        bar.next()

        self.vertical_profile = self.filling_ending(self.vertical_profile, coordinate_changes_allowed=False)
        self.roll = self.filling_ending(self.roll, coordinate_changes_allowed=False)
        self.track_width = self.filling_ending(self.track_width)

        bar.next()

        for dataIndex in range(len(self.files)):
            # Из-за особенности работы компьютера с числами с плавающей точкой может возникнуть ситуация, когда в self.coordinates[dataIndex]
            # сохранены почти одинаковые координаты (разница между ними порядка 1е-8).
            # Поэтому удостоверимся, что такой ситуации нет.
            # А если есть, тогда удалим повторяющуюся координату и данные, соответсвующее ей.
            try:
                double_infoIndex = np.where(np.isclose(np.diff(self.coordinates[dataIndex]), 0))[0][0]
                self.vertical_profile[dataIndex] = np.delete(self.vertical_profile[dataIndex], double_infoIndex)
                self.roll[dataIndex] = np.delete(self.roll[dataIndex], double_infoIndex)
                self.track_width[dataIndex] = np.delete(self.track_width[dataIndex], double_infoIndex)
                self.coordinates[dataIndex] = np.delete(self.coordinates[dataIndex], double_infoIndex)
            except IndexError:
                pass

        if self.testFlag:
            self.charting(self.coordinates, self.vertical_profile, x_axis1D=False, y_label='Перепад высот, мм',
                          title='Продолженные данные вертикального профиля', linewidth=2.5,
                          saved_dir=self.saved_dir, saved_name='Продолженные данные')

        # Тк у нас все self.coordinates[index] одинаковы, то нет смысла хранить их как список, поэтому сделаем приведение типа.
        self.coordinates = self.coordinates[0]

        self.charting(self.coordinates, self.vertical_profile,
                      linewidth=2.5,
                      label=self.files,
                      y_label='Перепад высот, мм',
                      title='Исходные данные для вертикального профиля',
                      saved_dir=self.saved_dir, saved_name='Сырой вертикальный профиль'
                      )

        self.get_common_marked_coordinates()
        bar.next()

        self.vertical_profile = self.tightening(self.vertical_profile, vertical_profileFlag=True)
        self.roll = self.tightening(self.roll)
        self.track_width = self.tightening(self.track_width)

        self.charting(self.coordinates, self.vertical_profile,
                      label=self.files,
                      title='Сведённый вертикальный профиль',
                      y_label='Перепад высот, мм',
                      saved_dir=self.saved_dir, saved_name='Сведённый вертикальный профиль')

        bar.next()

        # Заполним массив СКО
        self.std_array = np.array([np.std([self.vertical_profile[dataIndex][index] for dataIndex in range(len(self.vertical_profile))]) for index in range(len(self.coordinates))])

        self.charting(self.coordinates,
                      (self.std_array, ),
                      linewidth=3,
                      title=self.chart_info[self.filter_type]['Заголовок'],
                      saved_dir=self.saved_dir,
                      saved_name=self.chart_info[self.filter_type]['Имя файла'],
                      annotation=f'Количество файлов - {len(self.files)}\nКоличество точек сведения - {len(self.marked_coordinates)}'
                      )

        self.writing_to_CSV_file(self.vertical_profile, "Вертикальный профиль", self.file_namesInfo[self.filter_type])
        bar.next()

        self.get_rail_profile()
        bar.next()

        print('\nУспешное завершение программы\n')

    def get_rail_profile(self):
        profile_left = deepcopy(self.vertical_profile)
        profile_right = deepcopy(self.vertical_profile)

        for dataIndex in range(len(self.vertical_profile)):
            for index in range(len(self.vertical_profile[dataIndex])):
                profile_right[dataIndex][index] -= 0.47 * self.roll[dataIndex][index]
                profile_left[dataIndex][index] += (self.track_width[dataIndex][index] - 0.47) * self.roll[dataIndex][index]

        if self.testFlag:
            self.charting(self.coordinates, self.track_width, title='Ширина колеи после стяжки (мм)',
                          saved_dir=self.saved_dir, saved_name='Ширина колеи')
            self.charting(self.coordinates, self.roll, title='Крен', saved_dir=self.saved_dir, saved_name='Крен',
                          y_label='радианы * 0,001')

        # Усредним профиль правого и левого рельсов
        profile_left = np.array([np.mean([profile_left[dataIndex][index] for dataIndex in range(len(profile_left))])
                                 for index in range(len(profile_left[0]))])
        profile_right = np.array([np.mean([profile_right[dataIndex][index] for dataIndex in range(len(profile_right))])
                                  for index in range(len(profile_right[0]))])

        fig, ax = plt.subplots(nrows=1, ncols=2, tight_layout=True)

        fig.set_figheight(9)
        fig.set_figwidth(16)

        ax[0].plot(self.coordinates, profile_left * 1000, linewidth=3)
        ax[0].grid()
        ax[0].set_title('Вертикальный профиль левого рельса', weight='bold')
        ax[0].set_xlabel('Пройденный путь, м')
        ax[0].set_ylabel('Перепад высот, мм')

        ax[1].plot(self.coordinates, profile_right * 1000, linewidth=3, color='tab:red')
        ax[1].grid()
        ax[1].set_title('Вертикальный профиль правого рельса', weight='bold')
        ax[1].set_xlabel('Пройденный путь, м')

        fig.savefig(f'{self.dir}/Стяжка/Вертикальный профиль/Графики/Профиль рельсов.png')

        # Запишем профили рельсов в CSV файл
        csv_file = open(f'{self.dir}/Стяжка/Вертикальный профиль/Профиль рельсов.csv', 'w')

        # Напишем первую стоку
        csv_file.write('Путь Профиль_левый Профиль_правый\n')

        # Запишем все данные
        for index in range(len(self.coordinates)):
            csv_file.write(f'{self.to_csv_format(self.coordinates[index])}')
            csv_file.write(f' {self.to_csv_format(profile_left[index])}')
            csv_file.write(f' {self.to_csv_format(profile_right[index])}\n')

        csv_file.close()
