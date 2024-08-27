import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def charting_rolling_mean(filename, n, date):
    if not os.path.isdir(f'./Графики/{date}'):
        os.mkdir(f'./Графики/{date}')

    data = pd.read_csv(f'./DataFrames/{date}/{filename}.csv', sep=' ')

    ########################################
    # Построение графиков абсолютных величин
    ########################################

    plt.rcParams['font.size'] = '16'
    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    acceleration = np.sqrt(
        pow(data['acceleration_X'], 2) + pow(data['acceleration_Y'], 2) + pow(data['acceleration_Z'], 2))
    angular_velocity = np.sqrt(
        pow(data['angular_velocity_X'], 2) + pow(data['angular_velocity_Y'], 2) + pow(data['angular_velocity_Z'], 2))

    x_axis = np.linspace(n / 2, len(data.index) - 1, len(data.index))

    ax[0].plot(x_axis, acceleration.rolling(n).mean(), color='tab:blue')
    ax[0].set_ylabel('Величина ускорения', fontsize=14)
    ax[0].tick_params(bottom=False, labelbottom=False)
    ax[0].grid()

    ax[1].plot(x_axis, angular_velocity.rolling(n).mean(), color='tab:red')
    ax[1].set_ylabel('Величина угловой\nскорости', fontsize=14)
    ax[1].tick_params(bottom=False, labelbottom=False)
    ax[1].grid()

    plt.suptitle(f'Анализ данных из {filename}\n с помощью плавающего среднего для n = {n}', fontsize=18)
    fig.tight_layout()

    if not os.path.isdir(f'./Графики/{date}/Абсолютные величины'):
        os.mkdir(f'./Графики/{date}/Абсолютные величины')

    fig.savefig(f'./Графики/{date}/Абсолютные величины/{filename}_Charts.png')
    plt.close(fig=fig)

    ########################################
    # Построение графиков проекций ускорений
    ########################################

    if not os.path.isdir(f'./Графики/{date}/По осям'):
        os.mkdir(f'./Графики/{date}/По осям')

    fig, ax = plt.subplots(nrows=3, ncols=1)
    plt.rcParams['font.size'] = '16'
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axis_name = ['X', 'Y', 'Z']
    colors = ['tab:blue', 'tab:red', 'tab:green']

    for i in range(0, 3):
        ax[i].plot(x_axis, data[f'acceleration_{axis_name[i]}'].rolling(n).mean(), color=colors[i])
        ax[i].set_ylabel(f'Величина ускорения\n по оси {axis_name[i]}', fontsize=14)
        ax[i].tick_params(bottom=False, labelbottom=False)
        ax[i].grid()

    plt.suptitle(f'Анализ ускорений из {filename}\n с помощью плавающего среднего для n = {n}', fontsize=18)
    fig.tight_layout()

    if not os.path.isdir(f'./Графики/{date}/По осям/Ускорения'):
        os.mkdir(f'./Графики/{date}/По осям/Ускорения')

    fig.savefig(f'./Графики/{date}/По осям/Ускорения/{filename}_accelerations.png')
    plt.close(fig=fig)

    ################################################
    # Построение графиков проекций угловых скоростей
    ################################################

    fig, ax = plt.subplots(nrows=3, ncols=1)
    plt.rcParams['font.size'] = '16'
    fig.set_figheight(10)
    fig.set_figwidth(15)

    colors = ['tab:cyan', 'tab:orange', 'tab:purple']
    for i in range(0, 3):
        ax[i].plot(x_axis, data[f'angular_velocity_{axis_name[i]}'].rolling(n).mean(), color=colors[i])
        ax[i].set_ylabel(f'Величина угловой скорости\n по оси {axis_name[i]}', fontsize=14)
        ax[i].tick_params(bottom=False, labelbottom=False)
        ax[i].grid()

    plt.suptitle(f'Анализ угловых скоростей из {filename}\n с помощью плавающего среднего для n = {n}', fontsize=18)
    fig.tight_layout()

    if not os.path.isdir(f'./Графики/{date}/По осям/Угловые скорости'):
        os.mkdir(f'./Графики/{date}/По осям/Угловые скорости')

    fig.savefig(f'./Графики/{date}/По осям/Угловые скорости/{filename}_angular_velosities.png')
    plt.close(fig=fig)


n = 500  # Величина сдвига для плавающего среднего
# filenames = ['31.01.24_1_out', '31.01.24_2_in', '31.01.24_3_out', '31.01.24_4_in',
#              '31.01.24_5_out', '31.01.24_6_in', 'test_5mit']

filenames = ['1_test', '2_test']
for i in range(len(filenames)):
    charting_rolling_mean(filenames[i], n, '16.04.24')
    print(f'Графики значений из файла {filenames[i]} успешно сохранёны')
