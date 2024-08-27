import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


# -*- coding: utf-8 -*-


def clearing(column):
    arr = np.zeros(len(column))
    for j in range(len(column)):
        arr[j] = column[j].replace(",", ".")

    return arr


def approximate(x1, y1, x2, y2):
    return np.linalg.inv(np.array([[x1, 1], [x2, 1]])).dot(np.array([y1, y2]))


def interpolation(coordinates, x, y, h):
    x_array = np.zeros(1)
    y_array = np.zeros(1)
    h_array = np.zeros(1)
    distance_array = np.zeros(1)

    coord_min = np.ceil(coordinates[0])
    coord_max = np.floor(coordinates[-1])

    # Типа do-while
    coord = coord_min
    x_line = approximate(coordinates[0], x[0], coordinates[1], x[1])
    y_line = approximate(coordinates[0], y[0], coordinates[1], y[1])
    h_line = approximate(coordinates[0], h[0], coordinates[1], h[1])

    x_array[0] = x_line[0] * coord + x_line[1]
    y_array[0] = y_line[0] * coord + y_line[1]
    h_array[0] = h_line[0] * coord + h_line[1]
    distance_array[0] = coord

    coord += 1

    while coord <= coord_max:
        index = np.argmin(abs(coord - coordinates))
        index = index if coord > coordinates[index] else index - 1     # Определим с какой стороны находится coord относительно distance[index],
                                                                       # чтобы выбрать правильные точки для approximate

        x_line = approximate(coordinates[index], x[index], coordinates[index + 1], x[index + 1])
        y_line = approximate(coordinates[index], y[index], coordinates[index + 1], y[index + 1])
        h_line = approximate(coordinates[index], h[index], coordinates[index + 1], h[index + 1])

        x_array = np.append(x_array, x_line[0] * coord + x_line[1])
        y_array = np.append(y_array, y_line[0] * coord + y_line[1])
        h_array = np.append(h_array, h_line[0] * coord + h_line[1])
        distance_array = np.append(distance_array, coord)

        coord += 1

    return distance_array, x_array, y_array, h_array


file = './Питер/Ведомость координат стыков Питер.txt'

label, X, Y, H = np.loadtxt(file, dtype=str, skiprows=1, unpack=True)

X = (clearing(X)).astype(float)
Y = (clearing(Y)).astype(float)
H = (clearing(H)).astype(float)

X_right = X[46:]
Y_right = Y[46:]
H_right = H[46:]

X_left = X[:46]
Y_left = Y[:46]
H_left = H[:46]


# pprint(label_right)
# pprint(X_right)
# pprint(X_left)

coord_right = np.array(np.sqrt(X_right**2 + Y_right**2 + H_right**2))
coord_left = np.array(np.sqrt(X_left**2 + Y_left**2 + H_left**2))

# print(coord_right[0], coord_left[0])
# pprint(coord_right)
# print(coord_right[-1] - coord_right[0])

coord_right, X_right, Y_right, H_right = interpolation(coord_right, X_right, Y_right, H_right)
coord_left, X_left, Y_left, H_left = interpolation(coord_left, X_left, Y_left, H_left)

# pprint(coord_right)
# pprint(coord_left)
#
# print(len(coord_right), len(H_right))

plt.rcParams['font.size'] = '12'
fig_right, ax_right = plt.subplots()

ax_right.plot(coord_right - coord_right[0], -(H_right - H_right[0]) * 1000, color='tab:red', linewidth=3)

ax_right.grid()
ax_right.set_xlabel('Пройденный путь, м', fontsize=14)
ax_right.set_ylabel('Перепад высот, мм', fontsize=14)
ax_right.set_title('Профиль правый', weight='bold')

fig_left, ax_left = plt.subplots()

ax_left.plot(coord_left - coord_left[0], (H_left - H_left[0]) * 1000, color='tab:olive', linewidth=3)

ax_left.grid()
ax_left.set_xlabel('Пройденный путь, м', fontsize=14)
ax_left.set_ylabel('Перепад высот, мм', fontsize=14)
ax_left.set_title('Профиль левый', weight='bold')


csv_file_left = open('./Питер/Ведомость Питера интерполированная левый.csv', 'w')
csv_file_left.write('Номер Координата    X   Y   H\n')

for i in range(len(coord_left)):
    csv_file_left.write(f'{str(coord_left[i]).replace(".", ",")}00 {str(round(X_left[i], 6)).replace(".", ",")} '
                        f'{str(round(Y_left[i], 6)).replace(".", ",")} {str(round(H_left[i], 6)).replace(".", ",")}\n')

csv_file_left.close()

csv_file_right = open('./Питер/Ведомость Питера интерполированная правый.csv', 'w')
csv_file_right.write('Номер Координата    X   Y   H\n')

for i in range(len(coord_right)):
    csv_file_right.write(f'{str(coord_right[i]).replace(".", ",")}00 {str(round(X_right[i], 6)).replace(".", ",")} '
                         f'{str(round(Y_right[i], 6)).replace(".", ",")} {str(round(H_right[i], 6)).replace(".", ",")}\n')

csv_file_right.close()

fig_right.set_figheight(8)
fig_right.set_figwidth(12)
# fig_right.savefig('./Питер/Профиль правый.png')

plt.show()
