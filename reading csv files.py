import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

"""
Чтение csv файлов
"""


def clearing(pd_col):
    arr = np.zeros(len(pd_col))
    for i in range(len(pd_col)):
        arr[i] = pd_col[i].replace(",", ".")

    return arr


self_dir = './CSV files/17.07.24'
files = []
file_names = []

with os.scandir(self_dir) as items:
    for item in items:
        if ".csv" in item.name and item.is_file():
            file_names.append(item.name)
            files.append(self_dir + "/" + item.name)


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


for index in range(len(files)):

    data = pd.read_csv(files[index], sep=' ')

    moving = clearing(data["Путь"])
    course = clearing(data["Курс"])
    roll = clearing(data["Крен"])
    pitch = clearing(data["Тангаж"])

    x_axis = np.linspace(0, len(moving)-1, len(moving))
    plt.plot(x_axis, moving)
    ###############################
    ax_courses.plot(moving, course, label=file_names[index])
    ax_courses.set_ylabel('Курс')
    ax_courses.set_xlabel('Пройденный путь')

    ###############################
    ax_roll.plot(moving, roll, label=file_names[index])
    ax_roll.set_ylabel('Крен')
    ax_roll.set_xlabel('Пройденный путь')

    ###############################
    ax_pitch.plot(moving, pitch, label=file_names[index])
    ax_pitch.set_ylabel('Тангаж')
    ax_pitch.set_xlabel('Пройденный путь')

    ###############################

ax_courses.legend()
ax_courses.grid()

ax_roll.legend()
ax_roll.grid()

ax_pitch.legend()
ax_pitch.grid()

fig_courses.suptitle('Курс')
fig_courses.tight_layout()
# fig_courses.savefig('D:/Работа/CSV телеги/Графики/Совместные/Курс.png')

fig_roll.suptitle('Крен')
fig_roll.tight_layout()
# fig_roll.savefig('D:/Работа/CSV телеги/Графики/Совместные/Крен.png')

fig_pitch.suptitle('Тангаж')
fig_pitch.tight_layout()
# fig_pitch.savefig('D:/Работа/CSV телеги/Графики/Совместные/Тангаж.png')

plt.show()
