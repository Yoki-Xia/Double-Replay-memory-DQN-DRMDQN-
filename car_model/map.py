# !/usr/bin/python
# -*- coding:UTF-8 -*-

import math
import cv2
import numpy as np

def get_map0():
    '''获得取地图'''
    map_size = (100, 100)
    raw_occupancy = np.zeros(map_size)

    '''地图边缘'''
    for i in range(map_size[1]):
        raw_occupancy[0][i] = 1
        raw_occupancy[map_size[0] - 1][i] = 1
    for i in range(map_size[0]):
        raw_occupancy[i][0] = 1
        raw_occupancy[i][map_size[1] - 1] = 1

    put_wall(raw_occupancy, row=35, column=15, flag=1, num=30)
    put_wall(raw_occupancy, row=15, column=70, flag=0, num=30)
    put_wall(raw_occupancy, row=70, column=60, flag=1, num=20)
    # put_wall(raw_occupancy, row=22, column=8, flag=0, num=8)
    # put_wall(raw_occupancy, row=20, column=22, flag=0, num=10)

    agt1_pos = [5, 5]
    goal_pos = [80, 80]

    return raw_occupancy, agt1_pos, goal_pos, map_size

def get_map1():
    '''获得取地图'''
    map_size = (100, 100)
    raw_occupancy = np.zeros(map_size)

    '''地图边缘'''
    for i in range(map_size[1]):
        raw_occupancy[0][i] = 1
        raw_occupancy[map_size[0] - 1][i] = 1
    for i in range(map_size[0]):
        raw_occupancy[i][0] = 1
        raw_occupancy[i][map_size[1] - 1] = 1

    put_wall(raw_occupancy, row=50, column=30, flag=1, num=40)

    # put_wall(raw_occupancy, row=15, column=70, flag=0, num=30)
    # put_wall(raw_occupancy, row=70, column=60, flag=1, num=20)

    agt1_pos = [5, 5]
    goal_pos = [80, 80]

    return raw_occupancy, agt1_pos, goal_pos, map_size

def get_map2():
    '''获得取地图'''
    map_size = (50, 50)
    raw_occupancy = np.zeros(map_size)

    '''地图边缘'''
    for i in range(map_size[1]):
        raw_occupancy[0][i] = 1
        raw_occupancy[map_size[0] - 1][i] = 1
    for i in range(map_size[0]):
        raw_occupancy[i][0] = 1
        raw_occupancy[i][map_size[1] - 1] = 1

    put_wall(raw_occupancy, row=25, column=10, flag=1, num=30)


    agt1_pos = [3, 25]
    goal_pos = [40, 25]

    return raw_occupancy, agt1_pos, goal_pos, map_size

def get_map3():
    '''获得取地图'''
    map_size = (50, 50)
    raw_occupancy = np.zeros(map_size)

    '''地图边缘'''
    for i in range(map_size[1]):
        raw_occupancy[0][i] = 1
        raw_occupancy[map_size[0] - 1][i] = 1
    for i in range(map_size[0]):
        raw_occupancy[i][0] = 1
        raw_occupancy[i][map_size[1] - 1] = 1

    put_square(raw_occupancy, [35, 0], [40, 15])
    put_square(raw_occupancy, [21, 15], [26, 30])
    put_square(raw_occupancy, [9, 30], [14, 49])
    # put_square(raw_occupancy, [10, 10], [25, 15])

    agt1_pos = [5, 45]
    goal_pos = [45, 5]

    return raw_occupancy, agt1_pos, goal_pos, map_size

def get_map4():
    '''获得取地图'''
    map_size = (50, 50)
    raw_occupancy = np.zeros(map_size)

    '''地图边缘'''
    for i in range(map_size[1]):
        raw_occupancy[0][i] = 1
        raw_occupancy[map_size[0] - 1][i] = 1
    for i in range(map_size[0]):
        raw_occupancy[i][0] = 1
        raw_occupancy[i][map_size[1] - 1] = 1


    put_square(raw_occupancy, [12,12], [22, 17])

    put_square(raw_occupancy, [12, 32], [22, 37])

    put_square(raw_occupancy, [32, 22], [42, 27])

    # put_square(raw_occupancy, [1, 32], [15, 35])
    # put_square(raw_occupancy, [25, 32], [49, 35])



    agt1_pos = [5, 45]
    goal_pos = [45, 5]

    return raw_occupancy, agt1_pos, goal_pos, map_size

def put_square(occupacy, pos1, pos2):
    '''
    在图中绘制一个矩形
    :param pos1: 矩形左上角位置
    :param pos2: 矩形右下角位置
    :return:

    '''
    (m, n) = occupacy.shape
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    print(dx, dy)
    for i in range(pos1[0], min(pos1[0]+dx, m-1)):
        print(i)
        put_wall(occupacy, pos1[1], i, 0, dy)


def put_wall(occupacy, row, column, flag, num):
    '''
    在某行某列开始向行或者列放置墙（即将其值置 1）
    :param occupacy:
    :param row:
    :param column:
    :param flag: 横着放还是竖着放, flag=0 沿x轴平行放，flag=1 沿y轴平行放
    :param num: 表示要沿这里画多长的墙
    :return: none，因为输入的occupacy是引用，会随着函数内的改变而改变
    '''
    (m, n) = occupacy.shape
    if flag == 0:
        if m >= row + num:
            for i in range(row,row+num):
                occupacy[i][column] = 1
        else:
            for i in range(row, m):
                occupacy[i][column] = 1
    else:
        if n >= column + num:
            for i in range(column,column+num):
                occupacy[row][i] = 1
        else:
            for i in range(column, n):
                occupacy[row][i] = 1