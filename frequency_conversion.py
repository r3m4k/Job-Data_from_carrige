import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def approximate(x1, y1, x2, y2):
    # left_side = np.array([y1, y2])
    # right_side = np.array([[x1, 1], [x2, 1]])
    return np.linalg.inv(np.array([[x1, 1], [x2, 1]])).dot(np.array([y1, y2]))


def acceleration_conversion(filename, date, old_period, new_period):

    axes = ['X', 'Y', 'Z']
    converted_data = pd.DataFrame()

    data = pd.read_csv(f'./DataFrames/{date}/{filename}.csv', sep=' ')
    length = data.index[-1] + 1

    for index in range(len(axes)):
        acceleration = data[f'acceleration_{axes[index]}']
        convert_acceleration = list()
        line_coef = np.array([approximate(i * old_period, acceleration[i], (i + 1) * old_period, acceleration[i+1])
                              for i in range(0, length-1)])
        time = 0
        i = 0
        while (time < (length - 1) * old_period):
            convert_acceleration.append(i * new_period * line_coef[(i * new_period) // old_period, 0] + line_coef[(i * new_period) // old_period, 1])
            convert_angular_velocity.append(i * new_period * line_coef[(i * new_period) // old_period, 0] + line_coef[(i * new_period) // old_period, 1])

            i += 1
            time += new_period

        if (time == (length - 1) * old_period):
                convert_acceleration.append(i * new_period * line_coef[((i - 1) * new_period) // old_period, 0] +
                                            line_coef[((i - 1) * new_period) // old_period, 1])

        # Если time будет в точности равен (length - 1) * old_period, то ((i - 1) * new_period) // old_period
        # будет иметь значение равное len(line_coef). А последний индекс line_coef является len(line_coef - 1).
        # Таким образом, будет обращение к несуществующему элемемнты массива

        converted_data[f'acceleration_{axes[index]}'] = convert_acceleration

    converted_data.to_csv(f'./DataFrames/{date}/converted csv files/{filename}_converted.csv', sep=' ', index=False)


def angular_velocity_convertion(filename, date, old_period, new_period):
    axes = ['X', 'Y', 'Z']
    converted_data = pd.DataFrame()

    data = pd.read_csv(f'./DataFrames/{date}/{filename}.csv', sep=' ')
    length = data.index[-1] + 1

    for index in range(len(axes)):
        angular_velocity = data[f'angular_velocity_{axes[index]}']
        convert_angular_velocity = list()
        ang_vel_line_coef = np.array([approximate(i * old_period, angular_velocity[i], (i + 1) * old_period, angular_velocity[i+1])
                              for i in range(0, length-1)])
        time = 0
        i = 0
        while (time < (length - 1) * old_period):
            convert_angular_velocity.append(i * new_period * ang_vel_line_coef[(i * new_period) // old_period, 0] +
                                            ang_vel_line_coef[(i * new_period) // old_period, 1])
            i += 1
            time += new_period

        if (time == (length - 1) * old_period):
                convert_angular_velocity.append(i * new_period * ang_vel_line_coef[((i - 1) * new_period) // old_period, 0] +
                                            ang_vel_line_coef[((i - 1) * new_period) // old_period, 1])
        np.asarray(convert_angular_velocity, dtype=np.double)

        convert_angular_velocity = np.array([new_period * (convert_angular_velocity[i] + convert_angular_velocity[i+1]) / 2 ])


def conversion(filename, date, old_period, new_period):
    acceleration_conversion(filename, date, old_period, new_period)
    angular_velocity_convertion(filename, date, old_period, new_period)



old_period = 7
new_period = 5
angular_velocity_coef = old_period * 360 / 86164
conversion('test', '31.01.24', old_period, new_period)
