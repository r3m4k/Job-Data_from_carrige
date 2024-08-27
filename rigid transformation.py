import matplotlib.pyplot as plt
import numpy as np

"""
Подбор угла поворота, горизонтального и вертикального смещений для файлов одинаковой длины
"""

file = "test.csv"


def turning(angle, x_axis, y_axis):
    """
    Поворачивает x_axis и y_axis на заданный угол angle
    :param angle: угол поворота
    :param x_axis: данные по горизонтали
    :param y_axis: данные по вертикали
    :return: Повёрнутые точки (x_axis, y_axis)
    """

    angle *= np.pi / 180
    length = range(len(shifted_axis))

    rotation_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                [-np.sin(angle), np.cos(angle)]])
    vectors = np.array([[x_axis[i], y_axis[i]] for i in length])
    rotated_vectors = np.array([rotation_matrix @ vectors[i] for i in length])
    x_axis = np.array([rotated_vectors[i, 0] for i in length])
    y_axis = np.array([rotated_vectors[i, 1] for i in length])

    return x_axis, y_axis


def rotation(angle_min, angle_max, step):
    """
    Вычисляет угол поворота второго графика относительно первого путём
    нахождения наименьшего среднеквадратичного отклонения расстояний между точками графиков

    Т.е. при искомом угле поворота расстояния между точками будут минимально отличаться друг от друга

    :param angle_min: нижняя граница диапозона перебираемых углов (в градусах)
    :param angle_max: верхняя граница диапозона перебираемых углов (в градусах)
    :param step: шаг с которым идёт перебор
    :return: угол поворота второго графика относительно первого
    """

    global shifted_axis, shifted_value

    angle_min *= np.pi / 180
    angle_max *= np.pi / 180
    step *= np.pi / 180

    length = range(len(shifted_axis))

    angle = angle_min

    angles = np.zeros(1)
    diff_std = np.zeros(1)

    # Типа do-while

    rotation_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                [-np.sin(angle), np.cos(angle)]])
    vectors = np.array([[shifted_axis[i], shifted_value[i]] for i in length])
    rotated_vectors = np.array([rotation_matrix @ vectors[i] for i in length])
    shifted_axis_rotated = np.array([rotated_vectors[i, 0] for i in length])
    shifted_value_rotated = np.array([rotated_vectors[i, 1] for i in length])

    diff = np.zeros(len(source_axis))

    for i in length:
        diff[i] = np.sqrt(
            (source_axis[i] - shifted_axis_rotated[i]) ** 2 + (source_value[i] - shifted_value_rotated[i]) ** 2)

    angles[0] = angle
    diff_std[0] = np.std(diff)

    angle += step

    while angle <= angle_max:
        rotation_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                    [-np.sin(angle), np.cos(angle)]])
        vectors = np.array([[shifted_axis[i], shifted_value[i]] for i in length])
        rotated_vectors = np.array([rotation_matrix @ vectors[i] for i in length])

        shifted_axis_rotated = np.array([rotated_vectors[i, 0] for i in length])
        shifted_value_rotated = np.array([rotated_vectors[i, 1] for i in length])

        diff = np.zeros(len(source_axis))

        for i in length:
            diff[i] = np.sqrt(
                (source_axis[i] - shifted_axis_rotated[i]) ** 2 + (source_value[i] - shifted_value_rotated[i]) ** 2)

        angles = np.append(angles, angle)
        diff_std = np.append(diff_std, np.std(diff))

        angle += step

    return angles[np.argmin(diff_std)] * 180 / np.pi


def translation():
    """
    Вычисляет вертикальное и горизонтальное смещение второго графика относительно первого

    :return: горизонтальный сдвиг, вертикальный сдвиг
    """
    global shifted_axis, shifted_value, ax

    horizontal_shift = np.zeros(len(source_axis))
    vertical_shift = np.zeros(len(source_axis))

    for i in range(len(shifted_axis)):
        horizontal_shift[i] = shifted_axis[i] - source_axis[i]
        vertical_shift[i] = shifted_value[i] - source_value[i]

    horizontal_shift = np.mean(horizontal_shift)
    vertical_shift = np.mean(vertical_shift)

    shifted_axis -= horizontal_shift
    shifted_value -= vertical_shift

    return horizontal_shift, vertical_shift


source_axis, source_value, shifted_axis, shifted_value = np.loadtxt(file, dtype=np.double, skiprows=1, unpack=True)

# Перенесём начало координат в (source_axis[0], source_value[0])
# source_axis -= source_axis[0]
# shifted_axis -= source_axis[0]
# source_value -= source_value[0]
# shifted_value -= source_value[0]

start_point = [shifted_axis[0], shifted_value[0]]


fig, ax = plt.subplots()

ax.plot(source_axis, source_value, label='эталон')
ax.plot(shifted_axis, shifted_value, label='исходный')

rotation_angle = rotation(-30, 30, 1)
rotation_angle = rotation(round(rotation_angle * 0.95, 1), round(rotation_angle * 1.05, 1), 0.1)
rotation_angle = rotation(round(rotation_angle * 0.95, 2), round(rotation_angle * 1.05, 2), 0.01)
print(round(rotation_angle, 2))

shifted_axis, shifted_value = turning(rotation_angle, shifted_axis, shifted_value)

ax.plot(shifted_axis, shifted_value)

shifted_axis += start_point[0] - shifted_axis[0]
shifted_value += start_point[1] - shifted_value[0]

shift = translation()

ax.plot(shifted_axis, shifted_value, label='повёрнутый\nи сдвинутый', linestyle='-.')

print(f'Горизонтальный сдвиг --> {shift[0]}')
print(f'Вертикальный сдвиг   --> {shift[1]}')

ax.legend()
plt.show()
