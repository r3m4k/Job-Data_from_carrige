import os
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


self_dir = './CSV files/17.07.24'
file_pitch = str()
file_roll = str()


def start():

    get_files()

    if file_pitch == '':
        print('Файл с данными тангажа не найден')
        exit(1)

    if file_roll == '':
        print('Файл с данными крена не найден')
        exit(1)

    with open(file_pitch, 'r') as _file:
        titles = _file.readline().split(' ')[1:]
    titles[-1] = titles[-1][: -1]     # Убрали \n в конце

    x_axis = np.loadtxt(file_pitch, dtype=str, skiprows=1, unpack=True)[0]
    pitch = np.loadtxt(file_pitch, dtype=str, skiprows=1, unpack=True)[1:]
    roll = np.loadtxt(file_roll, dtype=str, skiprows=1, unpack=True)[1:]

    x_axis = clearing(x_axis)
    for index in range(len(pitch)):
        pitch[index] = clearing(pitch[index])
        roll[index] = clearing(roll[index])

    x_axis = x_axis.astype(int)
    pitch = -pitch.astype(np.double) * np.pi / 180
    roll = roll.astype(np.double) * np.pi / 180

    start_points = get_start_points(pitch)       # Координата начала движения в метрах
    end_points = get_end_points(pitch)           # Координата конца движения в метрах

    # print(start_points)
    # print(end_points)

    profiles = np.array([integrate(pitch[index], start_points[index] - 1, end_points[index] - 1) for index in range(len(pitch))])

    # Подгоним графики так, чтобы они совпадали в последней точке
    mean_line_coefficient = np.mean([profiles[column][-1] for column in range(len(pitch))]) / len(x_axis)

    line_coefficients = []
    for index in range(len(profiles)):
        line_coefficients.append(profiles[index][-1] / end_points[index])

    for index in range(len(profiles)):
        profile = np.asarray(profiles[index])
        profile -= np.array([(line_coefficients[index] - mean_line_coefficient) * x for x in range(start_points[index] - 1, end_points[index])])
        profiles[index] = profile

    profiles_right = np.array([profiles[index] / 10 - 0.76 * np.sin(roll[index]) for index in range(len(profiles))])
    profiles_left = np.array([profiles[index] / 10 + 0.76 * np.sin(roll[index]) for index in range(len(profiles))])
    
    # Заполним пустые места и среднеквадратичное отклонение

    std_left = []   # Элементы, которые будут учитываться при вычислении среднеквадратичного отклонения
    std_right = []
    for index in range(len(x_axis)):
        std_left.append([])
        std_right.append([])

    output_table = []
    for column in range(len(pitch)):
        output_array = np.zeros((len(x_axis), 2))
        for row in range(len(x_axis)):
            if (x_axis[row] >= start_points[column]) and (x_axis[row] <= end_points[column]):
                output_array[row][0] = profiles_left[column][row - start_points[column] + 1]
                output_array[row][1] = profiles_right[column][row - start_points[column] + 1]
                std_left[row].append(profiles_left[column][row - start_points[column] + 1])
                std_right[row].append(profiles_right[column][row - start_points[column] + 1])

            elif x_axis[row] < start_points[column]:
                output_array[row][0] = profiles_left[column][0]
                output_array[row][1] = profiles_right[column][0]
            elif x_axis[row] > end_points[column]:
                output_array[row][0] = profiles_left[column][-1]
                output_array[row][1] = profiles_right[column][-1]

        output_table.append(output_array)

    std_left_array = np.array([np.std(std_left[i]) for i in range(len(x_axis))])
    std_right_array = np.array([np.std(std_right[i]) for i in range(len(x_axis))])

    # Запишем полученные данные в csv файл
    if not os.path.exists(self_dir + "/Профиль"):
        os.mkdir(self_dir + "/Профиль")

    csv_file = open(f'{self_dir}/Профиль/{file_pitch[file_pitch.find("Тангаж") - 9: file_pitch.find("Тангаж") - 1]}_профиль.csv', 'w')

    csv_file.write("Путь")
    for title in titles:
        csv_file.write(f' {title}_профиль_левый {title}_профиль_правый')
    csv_file.write(' СКО_левый СКО_правый\n')

    for row in range(len(x_axis)):
        csv_file.write(f'{str(x_axis[row])},000')
        for column in range(len(output_table)):
            csv_file.write(f' {str(round(output_table[column][row, 0], 8)).replace(".", ",")} {str(round(output_table[column][row, 1], 8)).replace(".", ",")}')
        csv_file.write(f' {str(round(std_left_array[row], 8)).replace(".", ",")} {str(round(std_right_array[row], 8)).replace(".", ",")}\n')

    csv_file.close()

    fig_left, ax_left = plt.subplots(nrows=2, ncols=1,
                                     gridspec_kw={'width_ratios': [1],
                                                  'height_ratios': [3, 1]})

    fig_left.set_figheight(8)
    fig_left.set_figwidth(12)

    fig_right, ax_right = plt.subplots(nrows=2, ncols=1,
                                       gridspec_kw={'width_ratios': [1],
                                                    'height_ratios': [3, 1]})

    fig_right.set_figheight(8)
    fig_right.set_figwidth(12)

    for index in range(len(profiles_left)):
        ax_left[0].plot(x_axis, profiles_left[index] * 1000, label=titles[index])
        ax_right[0].plot(x_axis, profiles_right[index] * 1000, label=titles[index])

    ax_left[0].legend()
    ax_left[0].set_title('Профиль левый', weight='bold')
    ax_left[0].grid()

    ax_right[0].legend()
    ax_right[0].set_title('Профиль правый', weight='bold')
    ax_right[0].grid()

    ax_left[1].plot(x_axis, std_left_array * 1000)
    ax_left[1].set_title('Среднеквадратичное отклонение', weight='bold')
    ax_left[1].grid()
    ax_left[1].set_ylim(0, 8)

    ax_right[1].plot(x_axis, std_right_array * 1000)
    ax_right[1].set_title('Среднеквадратичное отклонение', weight='bold')
    ax_right[1].grid()
    ax_right[1].set_ylim(0, 8)

    # fig_left.tight_layout()
    # fig_right.tight_layout()

    fig, ax = plt.subplots()
    ax.plot(x_axis, np.array([np.mean([profiles[column][row] for column in range(len(profiles))]) for row in range(len(x_axis))]) / 10, label='Средний', linewidth=2.5)
    ax.plot(x_axis, np.array([np.mean([profiles_left[column][row] for column in range(len(profiles))]) for row in range(len(x_axis))]), label='Левый', linewidth=2.5)
    ax.plot(x_axis, np.array([np.mean([profiles_right[column][row] for column in range(len(profiles))]) for row in range(len(x_axis))]), label='Правый', linewidth=2.5)
    ax.grid()
    ax.legend()
    fig.tight_layout()

    fig_left.savefig(f'{self_dir}/Профиль/{file_pitch[file_pitch.find("Тангаж") - 9: file_pitch.find("Тангаж") - 1]}_профиль_левый.png')
    fig_right.savefig(f'{self_dir}/Профиль/{file_pitch[file_pitch.find("Тангаж") - 9: file_pitch.find("Тангаж") - 1]}_профиль_правый.png')

    plt.show()

    # np.save('./CSV files/17.07.24/Профиль/СКО_левый.npy', std_left_array)
    # np.save('./CSV files/17.07.24/Профиль/СКО_правый.npy', std_right_array)


def get_start_points(table):
    result = []
    for column in table:
        for row in range(len(column) - 1):
            if column[row] != column[row + 1]:
                result.append(row + 1)
                break

    return result


def get_end_points(table):
    result = []
    for column in table:
        for row in range(len(column) - 1, 1, -1):
            if column[row] != column[row - 1]:
                result.append(row + 1)
                break

    return result


def clearing(column):
    arr = np.zeros(len(column))
    for j in range(len(column)):
        arr[j] = column[j].replace(",", ".")

    return arr


def integrate(profiles, start_index, end_index):
    integrated_array = np.zeros(end_index - start_index + 1)
    for i in range(1, end_index - start_index + 1):
        for j in range(start_index, i + start_index + 1):
            integrated_array[i] += profiles[j]

    return integrated_array


def get_files():
    global file_pitch, file_roll
    with os.scandir(self_dir + '/Нормированные') as items:
        for item in items:
            if ".csv" in item.name and item.is_file():
                if "Тангаж" in item.name:
                    file_pitch = self_dir + '/Нормированные/' + item.name
                if "Крен" in item.name:
                    file_roll = self_dir + '/Нормированные/' + item.name


start()
