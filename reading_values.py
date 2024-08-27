import binascii
import numpy as np
import pandas as pd


names = ['acceleration_X', 'acceleration_Y', 'acceleration_Z', 'angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z']


def modified_additional_code(data):

    # Example
    # data = 16360 in decimal
    # Bin_data = 0b11111111101000
    # bin_clear_data = 11111111101000
    # res = 00000000010111 or 23 in decimal

    bin_data = bin(data - 1)
    bin_clear_data = ''
    flag = False

    for i in range (len(bin_data)):
        if flag:
            bin_clear_data += bin_data[i]
        if bin_data[i] == 'b':
            flag = True

    res = ''
    for i in range(len(bin_clear_data)):
        if bin_clear_data[i] == '1':
            res += '0'
        else: res += '1'

    return int(res, 2)


def hex_data_to_decimal(hex_data):
    decimal_data = int(hex_data, 16)
    sign_const = 49152      # binary 1100000000000000
    sign_bin_flag = decimal_data & sign_const      # sign_bin_flag is 1100000000000000 ("-") or 0000000000000000 ("+")
    if sign_bin_flag == sign_const:
        decimal_data = modified_additional_code(decimal_data) * (-1)    # "-"
    # If the number is positive, then we will not convert anything.
    return decimal_data


def mod_add_code(number):
    result = int(number, 16)
    sign_const = result >> 14
    if (sign_const == 3) :	# It means that result < 0 in modified additional code
        result &= ~(1 << 15)
        result &= ~(1 << 14)
        result -= 1
        for i in range (0, 14):
            result ^= 1 << i
        result *= -1
    return result


def reading_from_log_file(filename, shot_begining):

    data_table = pd.DataFrame()
    data = [list(), list(), list(), list(), list(), list()]

    ##########################

    # data[0] = 'acceleration_X'
    # data[1] = 'acceleration_Y'
    # data[2] = 'acceleration_Z'
    # data[3] = 'angular_velocity_X'
    # data[4] = 'angular_velocity_Y'
    # data[5] = 'angular_velocity_Z'

    ##########################

    file = open(filename, 'rb')
    hex_data = binascii.hexlify(file.read(1))
    while hex_data:
        if hex_data == shot_begining[0]:
            hex_data = binascii.hexlify(file.read(1))
            if hex_data == shot_begining[1]:
                hex_data = binascii.hexlify(file.read(1))
                if hex_data == shot_begining[2]:
                    hex_data = binascii.hexlify(file.read(1))
                    if hex_data == shot_begining[3]:

                        for i in range(0, 6):
                            low_bit = binascii.hexlify(file.read(1))
                            high_bit = binascii.hexlify(file.read(1))
                            match i:
                                # case 0: data[0].append(hex_data_to_decimal(high_bit + low_bit))
                                # case 1: data[1].append(hex_data_to_decimal(high_bit + low_bit))
                                # case 2: data[2].append(hex_data_to_decimal(high_bit + low_bit))
                                # case 3: data[3].append(hex_data_to_decimal(high_bit + low_bit))
                                # case 4: data[4].append(hex_data_to_decimal(high_bit + low_bit))
                                # case 5: data[5].append(hex_data_to_decimal(high_bit + low_bit))
                                case 0: data[0].append(mod_add_code(high_bit + low_bit))
                                case 1: data[1].append(mod_add_code(high_bit + low_bit))
                                case 2: data[2].append(mod_add_code(high_bit + low_bit))
                                case 3: data[3].append(mod_add_code(high_bit + low_bit))
                                case 4: data[4].append(mod_add_code(high_bit + low_bit))
                                case 5: data[5].append(mod_add_code(high_bit + low_bit))
                        hex_data = binascii.hexlify(file.read(5))

        hex_data = binascii.hexlify(file.read(1))

    for i in range(len(data)):
        data[i] = np.asarray(data[i])
        data_table[names[i]] = data[i].tolist()

    file.close()

    return data_table


# for i in range(1, 7):
#     if i % 2:
#         file_name = f'./Исходные файлы/31.01.24_{i}_out.log'
#         csvfile_way = f'./DataFrames/31.01.24/31.01.24_{i}_out.csv'
#     else:
#         file_name = f'./Исходные файлы/31.01.24_{i}_in.log'
#         csvfile_way = f'./DataFrames/31.01.24/31.01.24_{i}_in.csv'
#
#     if i != 2:
#         reading_from_log_file(file_name, [b'7e', b'11', b'ff', b'c9']).to_csv(csvfile_way, sep=' ', index=False)
#     else:
#         reading_from_log_file(file_name, [b'7e', b'11', b'79', b'3f']).to_csv(csvfile_way, sep=' ', index=False)
#
#     print(f'File "{file_name}" has been successfully completed')
#
# reading_from_log_file('./Исходные файлы/test_5mit.log', [b'7e', b'11', b'ff', b'c9']).to_csv('./DataFrames/31.01.24/test_5mit.csv', sep=' ', index=False)


# for i in range (1, 3):
#     file_name = f'./Исходные файлы/16.04.24/{i}_test.log'
#     csvfile_way = f'./DataFrames/16.04.24/{i}_test.csv'
#     reading_from_log_file(file_name, [b'7e', b'11', b'ff', b'c9']).to_csv(csvfile_way, sep=' ', index=False)
#     print(f'File "{file_name}" has been successfully completed')

file_name = './Исходные файлы/31.01.24/test_5mit.log'
csvfile_way = './DataFrames/31.01.24/test_5mit.csv'
reading_from_log_file(file_name, [b'7e', b'11', b'ff', b'c9']).to_csv(csvfile_way, sep=' ', index=False)
