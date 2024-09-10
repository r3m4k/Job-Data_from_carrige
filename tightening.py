# -*- coding: utf-8 -*-

import argparse
from vertical_profile import VerticalProfile
from horisontal_plan import HorizontalPlan

############# Создание и настройка парсера для анализа аргументов командной строки #############


parser = argparse.ArgumentParser(description='')        # Создадим парсер для анализа аргументов командной строки


# Добавим обязательный аргумент profile, который указывает, какой профиль анализировать: вертикальный или горизонтальный
parser.add_argument("profile", type=str, choices=['v', 'vertical', 'h', 'horizontal'],
                    help="Указание программе, какой профиль необходимо анализировать: вертикальный или горизонтальный.")

# Добавим в него необязательный аргумент end_height, который передадим в конструктор класса measuring в качестве параметра end_height.
# Значение по умолчанию - 0.
# Используется только для вертикального профиля
parser.add_argument("-eh", "--end_height", type=float, default=0,
                    help="Разница эталонных данных между первой и последней точками измерения. (Задаётся только для построения вертикального профиля)")


# Добавим флаг тестировочного режима
parser.add_argument("-tm", "--test_mode", type=bool, default=False,
                    help='Режим при котором будут созданы дополнительные графики,'
                         'которые поясняют шаги выполнения программы.'
                         ' ---> True  (1) - включён'
                         ' ---> False (0) - выключен (значение по умолчанию)')

# Добавим тип используемого фильтра
parser.add_argument("-f", "--filter_type", type=str, choices=["average_filter", "median_filter"], default="average_filter",
                    help="Тип используемого фильтра - медианный или усредняющий")

# Если в дальнейшем будет так, что гироскоп на тележках будет повёрнут по направлению движения, а не перпендикулярно ему,
# то нужно будет ввести дополнительный параметр из командной строки, указывающий на это
parser.add_argument("--gyroscope_rotation", type=bool, default=True,
                    help='Флаг поворота гироскопа относительно направления движения.')


########################################################################################################


self_dir = './'       # Директория в которой будет работать стягивание данных

if parser.parse_args().profile == 'v' or parser.parse_args().profile == 'vertical':
    VerticalProfile(self_dir, parser.parse_args().filter_type, parser.parse_args().end_height,
                    parser.parse_args().test_mode, parser.parse_args().gyroscope_rotation)

elif parser.parse_args().profile == 'h' or parser.parse_args().profile == 'horizontal':
    HorizontalPlan(self_dir, parser.parse_args().filter_type, parser.parse_args().end_height, parser.parse_args().test_mode)


########################################################################################################
