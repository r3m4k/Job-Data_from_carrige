import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Создаёт csv файл с шагом в координате в 0.2 метра
"""


def clearing(pd_col):
    arr = np.zeros(len(pd_col))
    for i in range(len(pd_col)):
        arr[i] = pd_col[i].replace(",", ".")

    return arr


def approximate(x1, y1, x2, y2):
    return np.linalg.inv(np.array([[x1, 1], [x2, 1]])).dot(np.array([y1, y2]))


def line_coefficient(coordinate):

    min_index = np.argmin(np.abs(coordinate - moving))

    if coordinate > moving[min_index]:
        line = [approximate(moving[min_index - 1], course[min_index - 1], moving[min_index], course[min_index]),
                approximate(moving[min_index - 1], roll[min_index - 1], moving[min_index], roll[min_index]),
                approximate(moving[min_index - 1], pitch[min_index - 1], moving[min_index], pitch[min_index])]

    else:
        line = [approximate(moving[min_index], course[min_index], moving[min_index + 1], course[min_index + 1]),
                approximate(moving[min_index], roll[min_index], moving[min_index + 1], roll[min_index + 1]),
                approximate(moving[min_index], pitch[min_index], moving[min_index + 1], pitch[min_index + 1])]

    return line


files = ["D:/Работа/Данные с телеги/CSV files/19.03.24/19.03.24_1_out.csv",
         "D:/Работа/Данные с телеги/CSV files/19.03.24/19.03.24_2_in_ver1.csv",
         "D:/Работа/Данные с телеги/CSV files/19.03.24/19.03.24_2_in_ver2.csv"]

csv_files = ["D:/Работа/Данные с телеги/CSV files/19.03.24/Подогнанные/19.03.24_1_out.csv",
             "D:/Работа/Данные с телеги/CSV files/19.03.24/Подогнанные/19.03.24_2_in_ver1.csv",
             "D:/Работа/Данные с телеги/CSV files/19.03.24/Подогнанные/19.03.24_2_in_ver2.csv"]

filenames = ["19.03.24_1_out", "19.03.24_2_in_ver1", "19.03.24_2_in_ver2"]

colors = ['tab:blue', 'tab:red', 'tab:green']

fig_courses, ax_courses = plt.subplots(nrows=1, ncols=1)
fig_courses.set_figheight(10)
fig_courses.set_figwidth(15)
plt.rcParams['font.size'] = '14'

fig_roll, ax_roll = plt.subplots(nrows=1, ncols=1)
fig_roll.set_figheight(10)
fig_roll.set_figwidth(15)
plt.rcParams['font.size'] = '14'

fig_pitch, ax_pitch = plt.subplots(nrows=1, ncols=1)
fig_pitch.set_figheight(10)
fig_pitch.set_figwidth(15)
plt.rcParams['font.size'] = '14'

step = 0.2

for index in range(len(files)):
    data = pd.read_csv(files[index], sep=' ')

    moving = clearing(data["Путь"])
    course = clearing(data["Курс"])
    roll = clearing(data["Крен"])
    pitch = clearing(data["Тангаж"])

    coord_min = np.floor(np.min(moving) * 10) / 10 + step      # Округлили в меньшую сторону
    coord_max = np.ceil(np.max(moving)) - step                 # Округлили в большую сторону

    coord = coord_min

    coord_array = np.zeros(1)
    course_array = np.zeros(1)
    roll_array = np.zeros(1)
    pitch_array = np.zeros(1)

    course_line, roll_line, pitch_line = line_coefficient(coord)

    coord_array[0] = coord
    course_array[0] = course_line[0] * coord + course_line[1]
    roll_array[0] = roll_line[0] * coord + roll_line[1]
    pitch_array[0] = pitch_line[0] * coord + pitch_line[1]

    coord += step

    while coord < coord_max:
        course_line, roll_line, pitch_line = line_coefficient(coord)
        course_array = np.append(course_array, course_line[0] * coord + course_line[1])
        roll_array = np.append(roll_array, roll_line[0] * coord + roll_line[1])
        pitch_array = np.append(pitch_array, pitch_line[0] * coord + pitch_line[1])

        coord_array = np.append(coord_array, coord)

        coord += step

    csv_file = open(csv_files[index], 'w')
    csv_file.write('Путь Курс Крен Тангаж\n')

    for arr_index in range(len(course_array)):
        csv_file.write(f'{coord_array[arr_index]:.1f} {course_array[arr_index]} {roll_array[arr_index]} {pitch_array[arr_index]}\n')

    csv_file.close()

    ###############################
    ax_courses.plot(coord_array, course_array, color=colors[index], label=filenames[index])
    ax_courses.set_ylabel('Курс')
    ax_courses.set_xlabel('Пройденный путь')

    ###############################
    ax_roll.plot(coord_array, roll_array, color=colors[index], label=filenames[index])
    ax_roll.set_ylabel('Крен')
    ax_roll.set_xlabel('Пройденный путь')

    ###############################
    ax_pitch.plot(coord_array, pitch_array, color=colors[index], label=filenames[index])
    ax_pitch.set_ylabel('Тангаж')
    ax_pitch.set_xlabel('Пройденный путь')

    ###############################

ax_courses.legend()
ax_courses.grid()

ax_roll.legend()
ax_roll.grid()

ax_pitch.legend()
ax_pitch.grid()

fig_courses.tight_layout()
fig_courses.savefig("D:/Работа/Данные с телеги/Графики/19.03.24/Подогнанные/Курс.png")

fig_roll.tight_layout()
fig_roll.savefig('D:/Работа/Данные с телеги/Графики/19.03.24/Подогнанные/Крен.png')

fig_pitch.tight_layout()
fig_pitch.savefig('D:/Работа/Данные с телеги/Графики/19.03.24/Подогнанные/Тангаж.png')

# plt.show()
