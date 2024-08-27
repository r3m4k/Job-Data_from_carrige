import numpy as np
import matplotlib.pyplot as plt


def name_of_file(path, extension):
    """
    :param path: Путь к файлу
    :param extension: Расширение файла
    :return: Имя файла без расширения и пути расположения
    """
    file = path
    for i in range(0, file.count('/')):
        file = file[file.find('/') + 1:]

    return file[:file.find(extension)]


def get_limitsIndex(lower_limit, upper_limit):
    lowIndex = 0
    while True:
        if longitude[lowIndex]:
            if longitude[lowIndex] > lower_limit:
                lowIndex += 1
            else:
                break
        else:
            lowIndex += 1

    upperIndex = lowIndex
    while True:
        if longitude[upperIndex]:
            if longitude[upperIndex] > upper_limit:
                upperIndex += 1
            else:
                break
        else:
            upperIndex += 1

    return lowIndex, upperIndex


cities_coordinates = {
                      # 'Москва': {'Широта':  55.7522,
                      #            'Долгота': 37.6156},

                      # 'Санкт-Петербург': {'Широта': 59.9386,
                      #                     'Долгота': 30.3141},

                      # 'Тверь': {'Широта': 56.8584,
                      #           'Долгота': 35.9006},

                      'Завидово': {'Широта': 56.528667,
                                   'Долгота': 36.525176},

                      'Решётниково': {'Широта': 56.443004,
                                      'Долгота': 36.558750}

                      # 'Ржев': {'Широта': 56.2624,
                      #          'Долгота': 34.3282}
                      }

latitude = np.zeros(0)      # широта
longitude = np.zeros(0)     # долгота
# time = np.zeros(0)          # время


path = "D:/Проги/CRT/measuring/MSK-SPB.log"
# path = "D:/Проги/CRT/measuring/test.log"

flag = True

file = open(path, "r")

while flag:
    line = file.readline()
    flag = False if line == "" else True

    if line[:6] == "$GPGLL":
        # print(line)
        latitude_startIndex = line.find(",")
        latitude_endIndex = line.find(",", latitude_startIndex + 1)

        if (latitude_endIndex - latitude_startIndex) != 1:
            latitude = np.append(latitude, float(line[latitude_startIndex + 1: latitude_endIndex]))

            longitude_startIndex = line.find(",", latitude_endIndex + 2)
            longitude_endIndex = line.find(",", longitude_startIndex + 1)

            longitude = np.append(longitude, float(line[longitude_startIndex + 1: longitude_endIndex]))

            time_startIndex = line.find(",", longitude_endIndex + 2)
            time_endIndex = line.find(",", time_startIndex + 1)

            # time = np.append(time, float(line[time_startIndex + 1: time_endIndex]))
        else:
            latitude = np.append(latitude, None)
            longitude = np.append(longitude, None)

# upper_limitIndex = np.argmin(abs(latitude - 57))

for index in range(len(latitude)):
    if latitude[index]:
        latitude[index] = latitude[index] // 100 + (latitude[index] - (latitude[index] // 100) * 100) / 60
        longitude[index] = longitude[index] // 100 + (longitude[index] - (longitude[index] // 100) * 100) / 60

    # latitude[index] = 5624.25936                                       ---> исходное значение
    # latitude[index] // 100 = 56                                        ---> градусы
    # latitude[index] - (latitude[index] // 100) * 100 = 24.25936        ---> минуты
    # (latitude[index] - (latitude[index] // 100) * 100) / 60 = 0.40432  ---> доли градусов


# print(latitude)
# print(longitude)
# print(time)


# low_limitIndex, upper_limitIndex = get_limitsIndex(36.5, 36.6)
low_limitIndex, upper_limitIndex = 0, -1
# print(low_limitIndex, upper_limitIndex)

fig, ax = plt.subplots()

ax.plot(longitude[low_limitIndex: upper_limitIndex], latitude[low_limitIndex: upper_limitIndex], linewidth=2.5)
ax.grid()
ax.set_xlabel('Долгота', fontsize=14)
ax.set_xlim(36.5, 36.6)

ax.set_ylabel('Широта', fontsize=14)
ax.set_ylim(56.43, 56.55)

for key in cities_coordinates.keys():
    ax.scatter(cities_coordinates[key]['Долгота'], cities_coordinates[key]['Широта'], label=key, linewidths=3)
    ax.text(cities_coordinates[key]['Долгота'], cities_coordinates[key]['Широта'], f'{key}', fontsize=14)

fig.tight_layout()

plt.show()

fig.savefig(f'{path[:path.find(name_of_file(path, ".log"))]}{name_of_file(path, ".log")}.png')

file.close()
