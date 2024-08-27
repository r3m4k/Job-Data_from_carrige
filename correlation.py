import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def minimal(n, elements):
    min_value = elements[0]
    min_index = 0
    for i in range(1, n):
        if elements[i] < min_value:
            min_value = elements[i]
            min_index = i

    return min_value, min_index


def cross_corelation(data):
    n_step = 10  # Шаг, с которым будем изучать взаимную кореляцию

    part_range = lenght // num_parts  # количество кадров данных на одном участке, кроме последнего
    n_range = part_range // n_step

    for num in range(0, num_parts):

        print(f'Обработка {num + 1}-го участка из {num_parts}')

        i_step = num * part_range  # сдвигаем индексы к следущему участку

        if ((num > 0) and (num < num_parts - 1)):  # если не первый и не последний участок
            n_range = part_range // n_step
            # n отходит от центра на всю длину диапозона

            corr_func = np.zeros(2 * n_range + 1)
            for n in range(-n_range, n_range + 1):
                # сложим разность элементов двух массивов
                for i in range(i_step - part_range, i_step + part_range + 1):
                    corr_func[n + n_range] += abs(data[0][i] - data[1][i + n * n_step])
        else:
            n_range = part_range // n_step // 2
            # n отходит от центра на половину всего диапозона только на первом и последнем участке

            corr_func = np.zeros(2 * n_range + 1)
            for n in range(-n_range, n_range + 1):
                # сложим разность элементов двух массивов
                for i in range(i_step - part_range // 2, i_step + part_range // 2 + 1):
                    corr_func[n + n_range] += abs(data[0][i] - data[1][i + n * n_step])

        ax_corr[num // (num_parts // 2)][num % (num_parts // 2)].plot(
            np.linspace(-n_range, n_range, 2 * n_range + 1) * n_step, corr_func)
        ax_corr[num // (num_parts // 2)][num % (num_parts // 2)].grid()

        ax_corr[num // (num_parts // 2)][num % (num_parts // 2)].tick_params(labelsize=12, left=False, labelleft=False)
        ax_corr[num // (num_parts // 2)][num % (num_parts // 2)].set_title(f'Участок {num + 1}', size=14)
        ax_corr[num // (num_parts // 2)][num % (num_parts // 2)].annotate(
            f'n_min ≈ {(np.argmin(corr_func) - n_range) * n_step}',
            xy=(0.5, 0.8), xycoords='axes fraction', size=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightgrey", ec="dimgrey", lw=2))


files = ['./DataFrames/16.04.24/1_test.csv',
         './DataFrames/16.04.24/2_test.csv']

headlines = ['acceleration_X', 'acceleration_Y', 'acceleration_Z', 'angular_velocity_X',
             'angular_velocity_Y', 'angular_velocity_Z']

y_label = ['Величина ускорения\nпо оси X', 'Величина ускорения\nпо оси Y', 'Величина ускорения\nпо оси Z',
           'Величина угловой скорости\nпо оси X', 'Величина угловой скорости\nпо оси Y',
           'Величина угловой скорости\nпо оси Z']

suptitle = ['Взаимная корреляция величины ускорения по оси X',
            'Взаимная корреляция величины ускорения по оси Y',
            'Взаимная корреляция величины ускорения по оси Z',
            'Взаимная корреляция величины угловой скорости по оси X',
            'Взаимная корреляция величины угловой скорости по оси Y',
            'Взаимная корреляция величины угловой скорости по оси Z']

file_names = ['Ускорение по оси X', 'Ускорение по оси Y', 'Ускорение по оси Z',
              'Угловая скорость по оси X', 'Угловая скорость по оси Y', 'Угловая скорость по оси Z']

num_parts = 8  # на сколько частей разобьём массив

for index in range(len(headlines)):
    print(suptitle[index])
    data = [pd.read_csv(files[0], sep=' ')[headlines[index]],
            pd.read_csv(files[1], sep=' ')[headlines[index]]]

    rolling_n = 500

    # Уберём Nan элементы из-за плавающего среднего, развернём второй массив и переведём в np.array
    data[0] = np.asarray(data[0].rolling(rolling_n).mean()[rolling_n:])
    data[1] = np.asarray(data[1].rolling(rolling_n).mean()[-1: rolling_n: -1])

    # Тестовый массив
    # n = 1000
    # data = [np.zeros(n), np.zeros(n)]
    # for j in range(0, n):
    #     if not (j // 100) % 2:
    #         data[0][j] = j % 100
    #     else:
    #         data[0][j] = 100 - (j % 100)
    #
    #     if j > 43:
    #         data[1][j] = data[0][j - 43]

    lenght, min_index = minimal(2, [len(data[i]) for i in range(len(data))])

    for i in range(len(data)):
        if i != min_index:
            data[i] = data[i][:lenght]

    plt.rcParams['font.size'] = '16'

    fig, ax = plt.subplots()
    fig.set_figheight(7.65)
    fig.set_figwidth(13.6)

    x_axis = np.linspace(0, lenght - 1, lenght)

    fig.suptitle('Данные из файлов 1_test.csv и 2_test.csv', fontweight='bold', size=18)
    ax.plot(x_axis, data[0], label='данные из файла 1_test.csv')
    ax.plot(x_axis, data[1], label=f'данные из файла 2_test.csv\n(зеркально отображённые)')

    # Добавим вертикальные линии на график
    for i in range(0, num_parts):
        ax.plot([i * lenght // num_parts, i * lenght // num_parts],
                [np.min([np.min(data[0]), np.min(data[1])]), np.max([np.max(data[0]), np.max(data[1])])],
                linestyle='--', linewidth=1.5, color='gray')

    ax.plot([lenght, lenght], [np.min([np.min(data[0]), np.min(data[1])]), np.max([np.max(data[0]), np.max(data[1])])],
            linestyle='--', linewidth=1.5, color='gray')

    ######################################

    ax.grid()
    ax.set_ylabel(ylabel=y_label[index])
    ax.legend(loc='upper right', fontsize=14)

    fig_corr, ax_corr = plt.subplots(2, num_parts // 2)
    fig_corr.set_figheight(8)
    fig_corr.set_figwidth(13.6)

    cross_corelation(data)

    fig_corr.suptitle(suptitle[index], fontweight='bold', size=18, y=0.99)
    ax_corr[0][0].annotate('Если минимум при n > 0, то первая функция опрежает вторую. '
                           'Если минимум при n < 0, то первая функция отстаёт от второй',
                           xy=(0.05, 1.15), xycoords='axes fraction', size=12,
                           bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="steelblue", lw=2))

    # plt.show()

    fig.savefig(f'./Графики/16.04.24/Взаимная кореляция/Совместные графики/{file_names[index]}.png')
    # fig_corr.savefig(f'./Графики/16.04.24/Взаимная кореляция/Взаимная корреляция по участкам/{file_names[index]}.png')

    print()
