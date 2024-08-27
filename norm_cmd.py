import os
import numpy as np
from time import sleep
from progress.bar import ChargingBar


titles = ['Курс', 'Крен', 'Тангаж', 'Vn', 'Ve', 'Vh', 'Широта', 'Долгота', 'Высота', 'Ax', 'Ay', 'Az']

self_dir = "./"
# file_names = os.listdir(self_dir)
file_names = []
files = []
step = 0.2

with os.scandir(self_dir) as items:
    for item in items:
        if ".csv" in item.name and item.is_file():
            file_names.append(item.name)
            files.append(self_dir + "/" + item.name)

# print(file_names)
# print(files)


def main():
    print('Данные, записанные в csv файлах:')
    print('\t1  ---> Курс\n'
          '\t2  ---> Крен\n'
          '\t3  ---> Тангаж\n'
          '\t4  ---> Vn\n'
          '\t5  ---> Ve\n'
          '\t6  ---> Vh\n'
          '\t7  ---> Широта\n'
          '\t8  ---> Долгота\n'
          '\t9  ---> Высота\n'
          '\t10 ---> Ax\n'
          '\t11 ---> Ay\n'
          '\t12 ---> Az\n')

    print('Введите индекс обрабатываемой величины (для завершения введите 0):\t', end='')

    try:
        index = int(input())
        if not index or (index < 0 or index > 12):
            print('Ошибка ввода')
            sleep(3)
            exit()
            return

    except ValueError:
        print('Ошибка ввода')
        sleep(3)
        exit()
        return

    start(index - 1)


def start(index):
    bar = ChargingBar('Файлов обработано: ', max=len(files))

    coordinates = []
    data = []

    for file in files:
        file_coordinates = clearing(np.loadtxt(file, dtype=str, skiprows=1, unpack=True, encoding="UTF-8")[0])
        file_data = clearing(np.loadtxt(file, dtype=str, skiprows=1, unpack=True, encoding="UTF-8")[index + 6])

        coord_min = np.floor(np.min(file_coordinates) * 10) / 10 + step      # Округлили в меньшую сторону
        coord_max = np.ceil(np.max(file_coordinates) * 10) / 10 - step       # Округлили в большую сторону

        # Проверим, что координаты .2, .4, .6, .8, а не .3 и тд
        if abs(((coord_min / step) % 1) > 0.00001):
            coord_min += round(step / 2, 1)

        coord = coord_min
        coord_array = np.zeros(1)
        file_data_array = np.zeros(1)

        line = line_coefficient(coord, file_coordinates, file_data)

        coord_array[0] = coord
        file_data_array[0] = line[0] * coord + line[1]

        coord += step

        while coord < coord_max:
            line = line_coefficient(coord, file_coordinates, file_data)
            coord_array = np.append(coord_array, round(coord, 1))
            file_data_array = np.append(file_data_array, round(line[0] * coord + line[1], 8))
            coord += step

        coordinates.append(coord_array)
        data.append(file_data_array)
        bar.next()

    common_coordinates = np.unique(np.sort(np.concatenate(coordinates)))

    try:
        csv_file_temp = open(f'{os.path.abspath(self_dir + "/" + titles[index] + "_temp")}.csv', 'w')
    except NameError:
        print('Ошибка создания csv файла. Убедитесь в правильности ввода пути сохранения')
        return

    for row in range(len(common_coordinates)):
        csv_file_temp.write(f'{round(common_coordinates[row], 1)}')

        for column in range(len(data)):
            if common_coordinates[row] in coordinates[column]:
                csv_file_temp.write(f' {data[column][np.argwhere(coordinates[column] == common_coordinates[row])][0][0]}')

            else:
                if row > (len(common_coordinates) / 2):
                    csv_file_temp.write(f' {round(data[column][-1], 8)}')
                else:
                    csv_file_temp.write(f' {round(data[column][0], 8)}')

        csv_file_temp.write('\n')

    csv_file_temp.close()
    try:
        csv_file_temp = open(f'{os.path.abspath(self_dir + "/" + titles[index] + "_temp")}.csv', 'r')
        if not os.path.exists(self_dir + "/Нормированные"):
            os.mkdir(self_dir + "/Нормированные")
        csv_file = open(f'{os.path.abspath(self_dir + "/Нормированные/" + file_names[0][:8] + "_" + titles[index])}.csv', 'w')
    except NameError:
        print('Ошибка создания csv файла. Убедитесь в правильности ввода пути сохранения')
        return

    csv_file.write('Путь')

    for name in file_names:
        csv_file.write(f' {name}')

    csv_file.write('\n')

    for row in range(len(common_coordinates)):
        read_line = csv_file_temp.readline()
        read_line = read_line[read_line.find(' ') + 1:].replace(".", ",")
        if round(common_coordinates[row] % 1, 1) == 0:
            csv_file.write(f'{str(common_coordinates[row])[: str(common_coordinates[row]).find(".")]},000 ')
            csv_file.write(read_line)

    csv_file_temp.close()
    csv_file.close()
    os.remove(f'{os.path.abspath(self_dir + "/" + titles[index] + "_temp")}.csv')


def clearing(pd_col):
    arr = np.zeros(len(pd_col))
    for i in range(len(pd_col)):
        arr[i] = pd_col[i].replace(",", ".")

    return arr


def approximate(x1, y1, x2, y2):
    return np.linalg.inv(np.array([[x1, 1], [x2, 1]])).dot(np.array([y1, y2]))


def line_coefficient(x_coordinate, x_axis, y_axis):

    min_index = np.argmin(np.abs(x_coordinate - x_axis))

    if x_coordinate > x_axis[min_index]:
        line = approximate(x_axis[min_index - 1], y_axis[min_index - 1], x_axis[min_index], y_axis[min_index])
    else:
        line = approximate(x_axis[min_index], y_axis[min_index], x_axis[min_index + 1], y_axis[min_index + 1])

    return line


main()
