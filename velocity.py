import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def integration(array, lenght, flag):
    int_arr = np.empty(lenght - 1, dtype=np.double)
    sym = 0
    for i in range(lenght - 1):
        sym += frame_time * (array[i] + array[i+1])/2
        int_arr[i] = sym
    if flag == 'int_arr':
        return int_arr
    elif flag == 'sym':
        return sym

def counting_velocity(filename, date):
    data = pd.read_csv(f'./DataFrames/{date}/{filename}.csv', sep=' ')
    acceleration_X = data['acceleration_X']
    acceleration_X /= 100
    lenght = len(acceleration_X)
    velocity = integration(acceleration_X, lenght, 'int_arr')
    #np.append(velocity, None)

    plt.rcParams['font.size'] = '16'
    fig, ax = plt.subplots()
    # fig.set_figheight(10)
    # fig.set_figwidth(20)

    time = np.linspace(0, lenght-2, lenght-1)
    time *= frame_time
    ax.plot(time, velocity, color='tab:blue', label='Скорость без учёта сдвига')
    ax.plot([time[0], time[-1]], [velocity[0], velocity[-1]], linestyle='--', color='tab:cyan')

    shift = (velocity[-1] - velocity[0]) / (time[-1] - time[0])
    for i in range(lenght - 1):
        velocity[i] -= shift * time[i]

    ax.plot(time, velocity, color='tab:green', label='Скорость с учётом сдвига')
    # ax.text(0, 1, f'Пройденный путь равен {integration(velocity, lenght - 1, "sym"):.2f} м\n'
    #               f'Коэффицент сдвига равен {shift:.3f} м/с^2',
    #         ha="center", va="center", rotation=0, size=16,
    #         bbox=dict(boxstyle="round,pad=0.3",
    #                   fc="lightblue", ec="steelblue", lw=2))
    ax.annotate(f'Время движения {time[-1]} c ({(time[-1]/60):.3f} м)\n'
                f'Пройденный путь равен {integration(velocity, lenght - 1, "sym"):.2f} м\n'
                f'Коэффицент сдвига равен {shift:.3f} м/с^2',
                xy = (0.02, 0.6), xycoords='axes fraction', size=16,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="steelblue", lw=2))
    ax.legend(loc='upper left')
    ax.set_title(f'Анализ скорости из  {filename}.csv\n')
    ax.set_ylabel('Величина скорости, м/с', fontsize=14)
    ax.set_xlabel('Время движения, с', fontsize=14)
    ax.grid()
    plt.show()
    fig.savefig(f'./Графики/31.01.24/Скорости/{filename}.png')
    print(f'График скорости из файла {filename}.csv сохранён')


frame_time = 0.005
counting_velocity('31.01.24_3_out', '31.01.24')
counting_velocity('31.01.24_4_in', '31.01.24')
counting_velocity('31.01.24_5_out', '31.01.24')
counting_velocity('31.01.24_6_in', '31.01.24')
