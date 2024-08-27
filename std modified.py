import matplotlib.pyplot as plt
import numpy as np

"""
Вычисление расстояний между точкой и прямой, образованной точками на другом графике с таким же и следующим индексом
"""

file_1 = {'path': "D:/Работа/Данные с телеги/CSV files/19.03.24/Подогнанные/19.03.24_1_out.csv"}
file_2 = {'path': "D:/Работа/Данные с телеги/CSV files/19.03.24/Подогнанные/19.03.24_2_in_ver1.csv"}


def name_of_file(path, extension):
    """
    :param path: Путь к файлу
    :param extension: Расширение файла
    :return: Имя файла без расширения и пути расположения
    """
    file = path
    for i in range(0, file.count('/')):
        file = file[file.find('/') + 1:]

    return file[:file.find(extension)]


def rolling_mean(arr, rolling_n):
    """
    Функция для рассчёта плавающего среднего
    :param arr: исходный массив
    :param rolling_n: параметр плавающего стреднего
    :return: массив с плавающим средним длины len(arr)
    """

    rolling_arr = np.zeros(len(arr))
    for i in range(0, len(arr)):
        if i < rolling_n - 1:
            rolling_arr[i] = None
        else:
            rolling_arr[i] = np.sum(arr[i - rolling_n + 1: i + 1]) / rolling_n

    return rolling_arr


def approximate(x1, y1, x2, y2):
    return np.linalg.inv(np.array([[x1, 1], [x2, 1]])).dot(np.array([y1, y2]))


def std_modified(x_axis, data_1, data_2):
    result = np.zeros(len(x_axis) - 1)
    for index in range(len(x_axis) - 1):
        line = approximate(x_axis[index], data_2[index], x_axis[index + 1], data_2[index + 1])
        result[index] = abs((line[0] * x_axis[index] - data_1[index] + line[1]) / np.sqrt(line[0]**2 + 1))

    return result


def charting(x_axis, course, roll, pitch):
    colors = ['tab:blue', 'tab:cyan', 'tab:olive']

    rolling_n = 100

    height = 10
    width = 15

    ###############################

    fig_courses, ax_courses = plt.subplots(nrows=1, ncols=1)
    plt.rcParams['font.size'] = '14'
    fig_courses.set_figheight(height)
    fig_courses.set_figwidth(width)

    ax_courses.plot(x_axis, course, color=colors[0], label='Значения отклонений')
    ax_courses.plot(x_axis, rolling_mean(course, rolling_n),
                    color='tab:red', linewidth=3, label=f'Скользящее среднее\nrolling_n = {rolling_n}')
    ax_courses.set_ylabel('Курс', fontsize=14)
    ax_courses.set_xlabel('Пройденный путь', fontsize=14)

    ax_courses.annotate(
            f'Средняя добавка = {np.mean(rolling_mean(course, rolling_n)[rolling_n:]):.5f}',
            xy=(0.5, 0.95), xycoords='axes fraction', size=14,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightgrey", ec="dimgrey", lw=2))

    ax_courses.legend()
    ax_courses.grid()

    fig_courses.tight_layout()
    fig_courses.savefig("D:/Работа/Данные с телеги/Графики/19.03.24/std modified/Курс.png")

    ###############################

    fig_roll, ax_roll = plt.subplots(nrows=1, ncols=1)
    plt.rcParams['font.size'] = '14'
    fig_roll.set_figheight(height)
    fig_roll.set_figwidth(width)

    ax_roll.plot(x_axis, roll, color=colors[1], label='Значения отклонений')
    ax_roll.plot(x_axis, rolling_mean(roll, rolling_n),
                 color='tab:red', linewidth=3, label=f'Скользящее среднее\nrolling_n = {rolling_n}')
    ax_roll.set_ylabel('Крен', fontsize=14)
    ax_roll.set_xlabel('Пройденный путь', fontsize=14)

    ax_roll.annotate(
            f'Средняя добавка = {np.mean(rolling_mean(roll, rolling_n)[rolling_n:]):.5f}',
            xy=(0.5, 0.95), xycoords='axes fraction', size=14,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightgrey", ec="dimgrey", lw=2))

    ax_roll.legend()
    ax_roll.grid()

    fig_roll.tight_layout()
    fig_roll.savefig('D:/Работа/Данные с телеги/Графики/19.03.24/std modified/Крен.png')

    ###############################

    fig_pitch, ax_pitch = plt.subplots(nrows=1, ncols=1)
    plt.rcParams['font.size'] = '14'
    fig_pitch.set_figheight(height)
    fig_pitch.set_figwidth(width)

    ax_pitch.plot(x_axis, pitch, color=colors[2], label='Значения отклонений')
    ax_pitch.plot(x_axis, rolling_mean(pitch, rolling_n),
                  color='tab:red', linewidth=3, label=f'Скользящее среднее\nrolling_n = {rolling_n}')
    ax_pitch.set_ylabel('Тангаж', fontsize=14)
    ax_pitch.set_xlabel('Пройденный путь', fontsize=14)

    ax_pitch.annotate(
            f'Средняя добавка = {np.mean(rolling_mean(pitch, rolling_n)[rolling_n:]):.5f}',
            xy=(0.5, 0.95), xycoords='axes fraction', size=14,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightgrey", ec="dimgrey", lw=2))

    ax_pitch.legend()
    ax_pitch.grid()

    fig_pitch.tight_layout()
    fig_pitch.savefig('D:/Работа/Данные с телеги/Графики/19.03.24/std modified/Тангаж.png')

    ###############################


file_1['coordinate'], file_1['course'], file_1['roll'], file_1['pitch'] = np.loadtxt(file_1['path'], dtype=np.double, skiprows=1, unpack=True)
file_2['coordinate'], file_2['course'], file_2['roll'], file_2['pitch'] = np.loadtxt(file_2['path'], dtype=np.double, skiprows=1, unpack=True)


file_1['min_index'] = np.where(file_1['coordinate'] == max(file_1['coordinate'][0], file_2['coordinate'][0]))[0][0]
file_1['max_index'] = np.where(file_1['coordinate'] == min(file_1['coordinate'][-1], file_2['coordinate'][-1]))[0][0]

file_2['min_index'] = np.where(file_2['coordinate'] == max(file_1['coordinate'][0], file_2['coordinate'][0]))[0][0]
file_2['max_index'] = np.where(file_2['coordinate'] == min(file_1['coordinate'][-1], file_2['coordinate'][-1]))[0][0]

# Срезом ['coordinate'][min_index: max_index] мы получим пересечение координат из двух файлов


charting(x_axis=file_1['coordinate'][file_1['min_index']: file_1['max_index'] - 1],

         course=std_modified(file_1['coordinate'][file_1['min_index']: file_1['max_index']],
                             file_1['course'][file_1['min_index']: file_1['max_index']],
                             file_2['course'][file_2['min_index']: file_2['max_index']]),

         roll=std_modified(file_1['coordinate'][file_1['min_index']: file_1['max_index']],
                           file_1['roll'][file_1['min_index']: file_1['max_index']],
                           file_2['roll'][file_2['min_index']: file_2['max_index']]),

         pitch=std_modified(file_1['coordinate'][file_1['min_index']: file_1['max_index']],
                            file_1['pitch'][file_1['min_index']: file_1['max_index']],
                            file_2['pitch'][file_2['min_index']: file_2['max_index']])
         )
