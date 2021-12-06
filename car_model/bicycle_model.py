# !/usr/bin/python
# -*- coding:UTF-8 -*-

import math
import cv2
import numpy as np
import map

class KinematicModel(object):
    def __init__(self, psi, v, f_len, r_len):
        '''
        自行车模型，简易版本
        map大小为 100 * 100
        :param x: 小车x轴重心
        :param y: 小车y轴重心
        :param psi: 小车偏航角
        :param v: 小车速度
        :param f_len: 前轮距离重心位置
        :param r_len: 后轮距离重心位置
        '''

        # 位置初始化
        self.occupancy, self.agt1_pos, self.goal_pos, self.map_size = map.get_map4()
        self.agt1_pos_init = self.agt1_pos.copy()
        self.goal_pos_init = self.goal_pos.copy()
        self.psi_init = psi
        self.psi = psi          # 小车偏航角 用的是弧度制
        self.max_delta = 15       # 设置最大转向角
        self.v = v              # 线速度
        self.delta = 0          # 转向角   最大为 max_delta
        self.a = 0              # 加速度， 目前先设置为0
        self.action_n = 3       # 一共三种动作，向左5°，向右5°，不动

        self.f_len = f_len      # 前轮距离重心距离
        self.r_len = r_len      # 后轮距离重心距离
        # 用于cv2显示的图像
        self.pic = np.ones((500, 500, 3))
        self.r = (f_len + r_len) / 2 + 0.1            # 小车的半径
        self.goal_r = 1
        self.dt = 0.001 * 100           # 小车刷新频率为多少毫秒

        self._270_angle = 3 / 2 * np.pi
        self._360_angle = 2 * np.pi
        self._180_angle = np.pi
        self._90_angle = 1 / 2 * np.pi

        self.theta_point = []
        self.theta_distance = []
        # self.angle_distance = np.zeros()
        self.range_occupancy = []           # 用于存储附件的障碍物


        self.last_distance = self.get_distance()        # 用于存储小车与目标之间上一时刻的距离

    def get_state(self):
        return self.agt1_pos, self.psi, self.v

    def step(self, action):
        '''
        更新位置：
        :param a: 车辆加速度
        :param delta: 车的转向角
        :param dt:  间隔时间
        :return:    大地位置，偏航角， 速度
        '''
        if action == 0 and self.delta > -self.max_delta:
            self.delta -= 5
        elif action == 1 and self.delta < self.max_delta:
            self.delta += 5
        else:
            pass

        delta = np.pi * (self.delta / 180)

        reward = 0
        done = False
        step_reward = -0.2
        collision_reward = -50
        goal_reward = 100
        angle_less_90_reward = 0.1
        angle_bigger_90_reward = -1
        arrive = False

        beta = math.atan((self.r_len / (self.r_len + self.f_len)) * math.tan(delta))

        self.agt1_pos[0] = self.agt1_pos[0] + self.v * math.cos(self.psi + beta) * self.dt
        self.agt1_pos[1] = self.agt1_pos[1] + self.v * math.sin(self.psi + beta) * self.dt

        self.psi = self.psi + (self.v / self.f_len) * math.sin(beta) * self.dt
        self.v = self.v + self.a * self.dt

        self.get_line()         # 获取激光

        # 修正 psi 使得其在 0 ~ 2pi 之间
        if self.psi < 0:
            self.psi += 2 * np.pi
        elif self.psi > 2*np.pi:
            self.psi -= 2 * np.pi

        # 判断是否发生碰撞
        if self.judge_collision():
            done = True
            reward += collision_reward
            arrive = False

        # now_distance = self.get_distance()

        # 通过判断是否有更加靠近目标来判断给与的步行惩罚大小
        # if now_distance < self.last_distance:
        #     reward -= step_reward
        #     self.last_distance = now_distance
        # else:
        #     reward += step_reward
        #     self.last_distance = now_distance

        # 通关判断是否朝向目标来设定回报值
        angle = abs(self.psi - self.get_aim_angle())
        if angle > np.pi:
            angle = self._360_angle - angle
        angle_reward = (np.pi/2 - angle)/(np.pi/2)
        if angle_reward > 0:
            angle_reward *= 0.1
        else:
            angle_reward *= 0.5                                 # 设置正反向回报是不同的
        reward = reward + angle_reward

        # 老版本的角度惩罚
        # angle = abs(self.psi - self.get_aim_angle())
        # if angle < self._90_angle:
        #     reward += angle_less_90_reward
        # else:
        #     reward += angle_bigger_90_reward

        # 新版本的角度惩罚


        # 为了防止智能体老转向，对转向也给予一定的惩罚
        # angle_judge = abs(delta)         # 最大调度为15°的时候为0.26
        # reward -= angle_judge * 0.1         # 0.026

        # 判断是否到达目标
        if self.arrive_aim():
            done = True
            reward += goal_reward
            # self.reset()
            arrive = True

        return reward, done, arrive

    def get_distance(self):
        '''用于获取智能体与目标之间的距离'''
        distance = ((self.goal_pos[0] - self.agt1_pos[0])**2 + (self.goal_pos[1] - self.agt1_pos[1])**2)**0.5
        return distance

    def arrive_aim(self):
        dis = ((self.goal_pos[0] - self.agt1_pos[0])**2 + (self.goal_pos[1] - self.agt1_pos[1])**2)**0.5
        if dis < self.goal_r + self.r:
            return True
        else:
            return False

    def judge_collision(self):
        '''
            判断是否发生碰撞
            目前的判断方式相当于用一个矩形将小车框起来了
        :return:
        '''
        x = int(self.agt1_pos[0] + self.r)
        y = int(self.agt1_pos[1] + self.r)
        if self.occupancy[x, y] == 1 or x < 0 or x > 49 or y < 0 or y > 49:
            return True

        x = int(self.agt1_pos[0] + self.r)
        y = int(self.agt1_pos[1] - self.r)
        if self.occupancy[x, y] == 1 or x < 0 or x > 49 or y < 0 or y > 49:
            return True

        x = int(self.agt1_pos[0] - self.r)
        y = int(self.agt1_pos[1] + self.r)
        if self.occupancy[x, y] == 1 or x < 0 or x > 49 or y < 0 or y > 49:
            return True

        x = int(self.agt1_pos[0] - self.r)
        y = int(self.agt1_pos[1] - self.r)
        if self.occupancy[x, y] == 1 or x < 0 or x > 49 or y < 0 or y > 49:
            return True

        return False


    def reset(self, arrive):
        '''
        重置场景
        :return:
        '''

        # 固定位置生成智能体
        # if not arrive:
        self.agt1_pos = self.agt1_pos_init.copy()
        self.psi = self.psi_init
        self.delta = 0
        # self.last_distance = self.get_distance()

        # 随机生成目标位置
        # if arrive:
        #     while True:
        #         self.goal_pos = [np.random.randint(1, 49), np.random.randint(1, 49)]
        #         if self.occupancy[self.goal_pos[0]+1, self.goal_pos[1]+1] == 0 and\
        #                             self.occupancy[self.goal_pos[0]-1, self.goal_pos[1]-1] == 0 and\
        #                              self.occupancy[self.goal_pos[0]+1, self.goal_pos[1]-1] == 0 and\
        #                                 self.occupancy[self.goal_pos[0]-1, self.goal_pos[1]+1] == 0:
        #             break



    def render(self, self_play=False):
        # 将100*100的场景映射到500*500的框内显示
        pic = np.ones((500,500,3))
        # pic = self.pic
        l = int(500 / self.map_size[0])
        w = int(500 / self.map_size[1])

        xx = int(self.agt1_pos[0] * l)
        yy = int(self.agt1_pos[1] * w)
        r = int(self.r * 500/self.map_size[0])

        l = int(500 / self.map_size[0])
        w = int(500 / self.map_size[1])

        # 画线
        star_point = (int((self.agt1_pos[0]) * l), int((self.agt1_pos[1]) * w))
        for min_point_theta in self.theta_point:
            end_point = (int(min_point_theta[0] * l), int(min_point_theta[1] * w))
            cv2.line(pic, star_point, end_point, (0, 127, 255), 2, 4)

        # 绘制agent1
        cv2.circle(img=pic, center=(xx, yy), radius=r, color=(1,0,0), thickness=-1)

        # 获取箭头坐标的位置，绘制箭头
        plus_x, plus_y = self.arrowed_line()
        cv2.arrowedLine(img=pic, pt1=(xx, yy),pt2=(xx+plus_x, yy+plus_y), color=(0,0,1), thickness=1)

        self.get_range_occupanccy()     # 更新智能机附近障碍物的数目


        # 绘制occupacy
        for i in range(self.map_size[0]):
            for o in range(self.map_size[1]):
                if self.occupancy[i][o]:
                    # l = int(500 / self.map_size[0])
                    # w = int(500 / self.map_size[1])
                    if [i, o] in self.range_occupancy:
                        cv2.rectangle(pic, (i * l, o * w), ((i + 1) * l, (o + 1) * w), color=(255, 255, 0), thickness=-1)
                    else:
                        cv2.rectangle(pic, (i * l, o * w), ((i + 1) * l, (o + 1) * w), color=(0, 0, 0), thickness=-1)
                        # cv2.circle(img=pic, center=(int(i * 500/self.map_size[0]), int(o * 500/self.map_size[1])),
                        #            radius=int(1 * 500/self.map_size[0]), color=(0, 0, 0), thickness=-1)


        # 绘制目标
        xx = int(self.goal_pos[0] * 500 / self.map_size[0])
        yy = int(self.goal_pos[1] * 500 / self.map_size[1])
        r = int(self.goal_r * 500 / self.map_size[0])
        cv2.circle(img=pic, center=(xx, yy), radius=r, color=(0, 1, 0), thickness=-1)



        # 显示
        cv2.imshow('car model', pic)
        if not self_play:
            cv2.waitKey(1)

    def count_distance(self, pos1, pos2):
        return np.sqrt(np.sum(np.square(pos1 - pos2)))

    def get_range_occupanccy(self):
        '''
        获取智能体周围 x 平方米范围内的障碍物
        :return:
        '''
        self.range_occupancy = []
        for i in range(self.map_size[0]):
            for o in range(self.map_size[1]):
                if self.occupancy[i, o] != 0:
                    pos2 = np.array([i, o])
                    # 若距离小于10，纳入需要判断距离的范畴
                    if self.count_distance(pos1=self.agt1_pos, pos2=pos2) < 10:
                        self.range_occupancy.append([i, o])
        return self.range_occupancy

    def get_k_and_b(self, theta):
        '''
        根据激光角度 theta 判断智能体发射的激光线方程
        :param theta: 激光角度
        :return:
        '''
        k = np.tan(theta/180*np.pi)
        y = self.agt1_pos[0]
        x = self.agt1_pos[1]
        #修正1
        x = self.agt1_pos[0]
        y = self.agt1_pos[1]
        #取智能体的中心点,  pos[0]:y   pos[1]:x
        b = y - k * x
        return k, b

    def get_line(self):
        self.grids = self.get_range_occupanccy()

        # 获取到需要检测的方格位置后，开始检测激光与方格是否有交叉
        # thetas = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180,
        #           195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]          # 检测角度
        # detect_angle = - int(self.psi / np.pi * 180) + 90
        # if detect_angle >= 360:
        #     detect_angle -= 360
        # elif detect_angle < 0:
        #     detect_angle += 360
        detect_angle = int(self.psi / np.pi * 180)
        # detect_angle = 0
        thetas = []
        angles = [10, 20, 30, 40, 50, 60, 70]
        # print(thetas)
        for i in angles[::-1]:
            if detect_angle - i < 0:
                thetas.append(detect_angle - i + 360)
            else:
                thetas.append(detect_angle - i)
        thetas.append(detect_angle)
        for i in angles:
            if detect_angle + i >= 360:
                thetas.append(detect_angle + i - 360)
            else:
                thetas.append(detect_angle + i)

        # print(thetas)
        self.theta_point = []
        self.theta_distance = []
        detect_distance = self.r / np.tan(5/180 * np.pi)
        for theta in thetas:
            k, b = self.get_k_and_b(theta=theta)
            x = self.agt1_pos[0]
            y = self.agt1_pos[1]
            min_point_theta = (x + detect_distance * np.cos(theta/180*np.pi), y + detect_distance * np.sin(theta/180*np.pi))
            # print('x, y', x, y)
            # print(np.cos(theta/180), np.sin(theta/180))
            # print('min point theta',min_point_theta)
            min_distance = detect_distance

            #四个特殊角度的情况
            if theta == 0:
                for grid in self.grids:
                    x_grid, y_grid = grid[0], grid[1]
                    if x_grid >= x - self.r and y_grid == int(y):
                        if x_grid - x < min_distance:
                            min_distance = x_grid - x
                            min_point_theta = [x_grid, y]

            elif theta == 180:
                for grid in self.grids:
                    x_grid, y_grid = grid[0], grid[1]
                    if x_grid + 1 <= x - self.r and y_grid == int(y):
                        if x - (x_grid + 1) < min_distance:
                            min_distance = x - (x_grid + 1)
                            min_point_theta = [x_grid + 1, y]

            elif theta == 90:
                for grid in self.grids:
                    x_grid, y_grid = grid[0], grid[1]
                    if x_grid == int(x) and y_grid >= y - self.r:
                        if y_grid - y < min_distance:
                            min_distance = y_grid - y
                            min_point_theta = [x, y_grid]

            elif theta == 270:
                for grid in self.grids:
                    x_grid, y_grid = grid[0], grid[1]
                    if x_grid == int(x) and y_grid + 1 <= y - self.r:
                        if y - (y_grid + 1) < min_distance:
                            min_distance = y - (y_grid + 1)
                            min_point_theta = [x, y_grid + 1]

            # 判断第一象限的情况
            elif 0 < theta < 90:
                # print('x, y :', x, y)
                for grid in self.grids:
                    x_grid, y_grid = grid[0], grid[1]
                    if x_grid >= x and y_grid + 1 >= y:
                        # 情况3 与直线 x=x_grid的交点：
                        touch = False                           # 如果touch = True表示已经碰到方格，就不可能碰到方格其他地方了，后面就不用检测了
                        point_theta = None
                        y_junction = k * x_grid + b
                        if y_grid <= y_junction <= y_grid + 1:
                            point_theta = [x_grid, y_junction]
                            # print('2 : ', x_grid, y_grid)

                            touch = True
                        # 情况0 与直线 y=y_grid的交点
                        if touch == False:
                            x_junction = (y_grid - b) / k
                            # print(x_junction, y_grid)
                            if x_grid <= x_junction <= x_grid + 1:
                                point_theta = [x_junction, y_grid]
                                # print('1 : ', x_grid, y_grid)

                        # 当point_theta存在的时候进行取最小
                        if point_theta != None:
                            # print(point_theta)
                            distance = self.count_distance(pos1=np.array((x, y)), pos2=np.array(point_theta))
                            if distance < min_distance:
                                min_distance = distance
                                min_point_theta = point_theta

            # 判断第二象限的情况
            elif 90 < theta < 180:
                # print('x, y :', x, y)
                for grid in self.grids:
                    x_grid, y_grid = grid[0], grid[1]
                    if x_grid + 1 <= x and y_grid + 1  >= y:
                        # 情况0 与直线y=y_grid的交点：
                        touch = False                           # 如果touch = True表示已经碰到方格，就不可能碰到方格其他地方了，后面就不用检测了
                        point_theta = None
                        x_junction = (y_grid - b) / k
                        if x_grid <= x_junction <= x_grid + 1:
                            point_theta = [x_junction, y_grid]
                            # print('2 : ', x_grid, y_grid)
                            touch = True
                        # 情况1 与直线 x=x_grid+1的交点
                        if touch == False:
                            y_junction = k * (x_grid + 1) + b
                            # print(x_junction, y_grid)
                            if y_grid <= y_junction <= y_grid + 1:
                                point_theta = [x_grid + 1, y_junction]
                                # print('1 : ', x_grid, y_grid)
                        # 当point_theta存在的时候进行取最小
                        if point_theta != None:
                            # print(point_theta)
                            distance = self.count_distance(pos1=np.array((x, y)), pos2=np.array(point_theta))
                            if distance < min_distance:
                                min_distance = distance
                                min_point_theta = point_theta

            # 判断第三象限的情况
            elif 180 < theta < 270:
                # print('x, y :', x, y)
                for grid in self.grids:
                    x_grid, y_grid = grid[0], grid[1]
                    if x_grid + 1 <= x and y_grid <= y:
                        # 情况2 与直线y=y_grid+1的交点：
                        touch = False                           # 如果touch = True表示已经碰到方格，就不可能碰到方格其他地方了，后面就不用检测了
                        point_theta = None
                        x_junction = (y_grid + 1 - b) / k
                        if x_grid <= x_junction <= x_grid + 1:
                            point_theta = [x_junction, y_grid+1]
                            # print('2 : ', x_grid, y_grid)
                            touch = True
                        # 情况1 与直线 x=x_grid+1的交点
                        if touch == False:
                            y_junction = k * (x_grid + 1) + b
                            # print(x_junction, y_grid)
                            if y_grid <= y_junction <= y_grid + 1:
                                point_theta = [x_grid + 1, y_junction]
                                # print('1 : ', x_grid, y_grid)
                        # 当point_theta存在的时候进行取最小
                        if point_theta != None:
                            # print(point_theta)
                            distance = self.count_distance(pos1=np.array((x, y)), pos2=np.array(point_theta))
                            if distance < min_distance:
                                min_distance = distance
                                min_point_theta = point_theta

            # 判断第四象限的情况
            elif 270 < theta < 360:
                # print('x, y :', x, y)
                for grid in self.grids:
                    x_grid, y_grid = grid[0], grid[1]
                    if x_grid >= x and y_grid <= y:
                        # 情况2 与直线y=y_grid+1的交点：
                        touch = False                           # 如果touch = True表示已经碰到方格，就不可能碰到方格其他地方了，后面就不用检测了
                        point_theta = None
                        x_junction = (y_grid + 1 - b) / k
                        if x_grid <= x_junction <= x_grid + 1:
                            point_theta = [x_junction, y_grid+1]
                            # print('2 : ', x_grid, y_grid)
                            touch = True
                        # 情况3 与直线 x=x_grid的交点
                        if touch == False:
                            y_junction = k * (x_grid) + b
                            # print(x_junction, y_grid)
                            if y_grid <= y_junction <= y_grid + 1:
                                point_theta = [x_grid, y_junction]
                                # print('1 : ', x_grid, y_grid)
                        # 当point_theta存在的时候进行取最小
                        if point_theta != None:
                            # print(point_theta)
                            distance = self.count_distance(pos1=np.array((x, y)), pos2=np.array(point_theta))
                            if distance < min_distance:
                                min_distance = distance
                                min_point_theta = point_theta

            # min_point_theta = (min_point_theta[1], min_point_theta[0])
            self.theta_point.append(min_point_theta)
            self.theta_distance.append(min_distance)

    def arrowed_line(self):
        plus_x = int(20*math.cos(self.psi))
        plus_y = int(20*math.sin(self.psi))
        # plus_x = int(20 * math.cos(120/180*np.pi))
        # plus_y = int(20 * math.sin(120/180*np.pi))
        return plus_x, plus_y

    def get_observations(self):
        ''' 返回的值用于给神经网络训练数据 '''
        theta_distance = self.theta_distance / (self.r / np.tan(5/180 * np.pi))             # 激光距离，激光能探测的数据与小车自身大小有关
        relative_distance = (np.array(self.goal_pos) - np.array(self.agt1_pos)) / np.array(self.map_size)
        # relative_distance = self.get_distance()/np.array(self.map_size)                                              # 目标与小车的距离
        # aim_angle = abs(self.psi - self.get_aim_angle())
        # if aim_angle > np.pi:
        #     aim_angle = self._360_angle - aim_angle
        # aim_angle = aim_angle/np.pi
        # delta_psi = np.array([self.delta / self.max_delta, self.psi / (2*np.pi)])
        delta_psi = np.array([self.delta / self.max_delta, self.psi/(2*np.pi)])                      # 轮胎转向角+自身角度
        observation = np.hstack([theta_distance, relative_distance, delta_psi])             # [激光信息，相对距离，轮胎转向角，自身角度  ]
        return observation

    def get_aim_angle(self):
        '''
        获取目标相对于智能体的位置
        但要求处智能体方向是否是目标还需要用智能体方向着这个角度进行对比。
        :return:
        '''
        relative_distance = np.array(self.goal_pos) - np.array(self.agt1_pos)
        R = (relative_distance[0]**2 + relative_distance[1]**2) ** 0.5
        cos_angle = relative_distance[0] / R
        if relative_distance[1] >= 0:
            angle = np.arccos(cos_angle)
        else:
            angle = np.arccos(cos_angle)
            angle = 2 * np.pi - angle
        return angle




def play_env(model):
    ''' 用键盘操作单个智能体运动 '''
    total_reward = 0
    step_count = 0
    while True:
        model.render(self_play=True)
        actions = cv2.waitKey(1)
        if actions != -1:
            if actions == ord('a'):
                reward, done, arrive = model.step(action=0)
            elif actions == ord('d'):
                reward, done, arrive = model.step(action=1)
        else:
            reward, done, arrive = model.step(action=2)

        total_reward += reward
        step_count += 1
        angle = abs(model.psi - model.get_aim_angle())
        if angle > np.pi:
            angle = model._360_angle - angle
        # print(model.get_observations(), reward, angle/np.pi, model.get_observations().shape)
        # print(model.theta_distance)
        if done:
            # print(model.agt1_pos)
            model.reset(arrive)
            break
    return total_reward, step_count, arrive


