import os
import numpy as np
import matplotlib.pyplot as plt
# from pprint import pprint

coefficients = {'Тангаж': -np.pi / 180,
                'Крен': np.pi / 180,
                'Курс': np.pi / 180}

self_dir = "./CSV files/17.07.24"
file_names = []
files = []


def start():

    get_files()
    # pprint(files)

    for file in files:
        with open(file, 'r') as _file:
            titles = _file.readline().split(' ')[1:]
        titles[-1] = titles[-1][: -1]     # Убрали \n в конце

        x_axis = np.loadtxt(file, dtype=str, skiprows=1, unpack=True)[0]
        data = np.loadtxt(file, dtype=str, skiprows=1, unpack=True)[1:]

        x_axis = clearing(x_axis)
        for i in range(len(data)):
            data[i] = clearing(data[i])

        x_axis = x_axis.astype(int)
        data = data.astype(np.double)

        for key in coefficients.keys():
            if key in file:
                for index in range(len(data)):
                    data[index] *= coefficients[key]

        fig_raw, ax_raw = plt.subplots()
        fig_raw.set_figheight(9)
        fig_raw.set_figwidth(16)

        for index in range(len(data)):
            ax_raw.plot(x_axis, data[index], label=f'{titles[index]}_raw')

        ax_raw.legend()
        ax_raw.grid()
        ax_raw.set_title(f'{file_names[files.index(file)]} исходные данные', weight='bold')
        fig_raw.tight_layout()
        fig_raw.savefig(f'{self_dir}/Интегрированные/Графики/{file_names[files.index(file)]}_исходные.png')

        start_points = get_start_points(data)       # Координата начала движения в метрах
        end_points = get_end_points(data)           # Координата конца движения в метрах

        print(start_points)
        print(end_points)

        integrated_arrays = []

        fig_integrated, ax_integrated = plt.subplots()

        for index in range(len(data)):
            integrated_arrays.append(integrate(data[index], start_points[index] - 1, end_points[index] - 1))
            ax_integrated.plot(np.linspace(start_points[index], end_points[index], end_points[index] - start_points[index] + 1), integrated_arrays[index],
                               label=f'{titles[index]}')

        # Сместим по вертикали графики из файлов, которые были записаны не на всём участке

        full_columns = []   # Индексы столбцов, полученные из файлов, данные в которых начинаются с минимальной координаты
        min_coord = np.min(start_points)
        for column in range(len(integrated_arrays)):
            if min_coord == start_points[column]:
                full_columns.append(column)

        for index in range(len(integrated_arrays)):
            if index not in full_columns:
                integrated_arrays[index] += np.mean(np.array([integrated_arrays[column][start_points[index]] for column in range(len(integrated_arrays))]))
                ax_integrated.plot(np.linspace(start_points[index], end_points[index], end_points[index] - start_points[index] + 1), integrated_arrays[index],
                                   label=f'{titles[index]}_смещённый')

        mean_line_coefficient = np.mean([integrated_arrays[column][-1] for column in range(len(data))]) / len(x_axis)
        ax_integrated.plot(x_axis, np.array([mean_line_coefficient * (x - 1) for x in x_axis]), linestyle='--', color='tab:olive')

        ax_integrated.legend()
        ax_integrated.grid()
        ax_integrated.set_title(f"{file_names[files.index(file)]} интегрированные", weight='bold')
        fig_integrated.tight_layout()

        # Подгоним графики так, чтобы они совпадали в последней точке

        line_coefficients = []
        for index in range(len(integrated_arrays)):
            line_coefficients.append(integrated_arrays[index][-1] / end_points[index])

        for index in range(len(integrated_arrays)):
            integrated_array = np.asarray(integrated_arrays[index])
            integrated_array -= np.array([(line_coefficients[index] - mean_line_coefficient) * x for x in range(start_points[index] - 1, end_points[index])])
            integrated_arrays[index] = integrated_array

        # Заполним пустые места и среднеквадратичное отклонение

        std_elements = []   # Элементы, которые будут учитываться при вычислении среднеквадратичного отклонения
        for index in range(len(x_axis)):
            std_elements.append([])

        output_table = []
        for column in range(len(data)):
            output_array = np.zeros(len(x_axis))
            for row in range(len(x_axis)):
                if (x_axis[row] >= start_points[column]) and (x_axis[row] <= end_points[column]):
                    output_array[row] = integrated_arrays[column][row - start_points[column] + 1]
                    std_elements[row].append(integrated_arrays[column][row - start_points[column] + 1])
                elif x_axis[row] < start_points[column]:
                    output_array[row] = integrated_arrays[column][0]
                elif x_axis[row] > end_points[column]:
                    output_array[row] = integrated_arrays[column][-1]
            output_table.append(output_array)

        std_array = np.array([np.std(std_elements[i]) for i in range(len(x_axis))])

        # Запишем полученные данные в csv файл
        if not os.path.exists(self_dir + "/Интегрированные"):
            os.mkdir(self_dir + "/Интегрированные")

        csv_file = open(f'{self_dir}/Интегрированные/{file_names[0][: file_names[0].find(".csv")]}_интегрированный.csv', 'w')

        csv_file.write("Путь")
        for title in titles:
            csv_file.write(f' {title}')
        csv_file.write(' Среднеквадратичное отклонение\n')

        for row in range(len(x_axis)):
            csv_file.write(f'{str(x_axis[row])},000')
            for column in range(len(output_table)):
                csv_file.write(f' {str(round(output_table[column][row], 8)).replace(".", ",")}')
            csv_file.write(f' {str(round(std_array[row], 8)).replace(".", ",")}\n')

        csv_file.close()

        # if "Тангаж" in file:
        #     np.save('./CSV files/17.07.24/Интегрированные/СКО.npy', std_array)

        fig, ax = plt.subplots(nrows=2, ncols=1,
                               gridspec_kw={'width_ratios': [1],
                                            'height_ratios': [3, 1]})

        fig.set_figheight(9)
        fig.set_figwidth(16)

        for index in range(len(output_table)):
            ax[0].plot(x_axis, output_table[index], label=titles[index])

        ax[0].legend()
        ax[0].grid()
        ax[0].set_title(f"{file_names[files.index(file)]} интегрированные и совмещённые", weight='bold')

        ax[1].plot(x_axis, std_array)
        ax[1].legend()
        ax[1].grid()
        ax[1].set_title('Среднеквадратичное отклонение', weight='bold')
        fig.tight_layout()
        fig.savefig(f'{self_dir}/Интегрированные/Графики/{file_names[files.index(file)]}_интегрированные.png')

        plt.show()


def integrate(data, start_index, end_index):
    integrated_array = np.zeros(end_index - start_index + 1)
    for i in range(1, end_index - start_index + 1):
        for j in range(start_index, i + start_index + 1):
            integrated_array[i] += data[j]

    return integrated_array


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


def get_files():
    with os.scandir(self_dir + '/Нормированные') as items:
        for item in items:
            if ".csv" in item.name and item.is_file():
                file_names.append(item.name)
                files.append(self_dir + "/Нормированные/" + item.name)


def clearing(column):
    arr = np.zeros(len(column))
    for j in range(len(column)):
        arr[j] = column[j].replace(",", ".")

    return arr


start()
