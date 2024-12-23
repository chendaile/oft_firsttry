import random
import numpy as np

# 定义常量
NUM_CITIES = 10 # 城市数量
NUM_ANTS = 100   # 蚂蚁数量
ALPHA = 1       # 信息素的重要性
BETA = 1        # 启发式信息的重要性
RHO = 0.5      # 信息素挥发因子
Q = 100         # 信息素强度
ITERATIONS = 100 # 迭代次数

# 距离矩阵（4个城市之间的距离）
distances = np.array([
    [0, 29, 20, 21, 17, 28, 14, 24, 21, 15],  # 城市 0 到 其他城市的距离
    [29, 0, 15, 17, 12, 23, 18, 22, 13, 18],  # 城市 1 到 其他城市的距离
    [20, 15, 0, 28, 18, 20, 21, 17, 15, 25],  # 城市 2 到 其他城市的距离
    [21, 17, 28, 0, 13, 28, 15, 12, 14, 17],  # 城市 3 到 其他城市的距离
    [17, 12, 18, 13, 0, 15, 22, 13, 17, 10],  # 城市 4 到 其他城市的距离
    [28, 23, 20, 28, 15, 0, 27, 26, 23, 20],  # 城市 5 到 其他城市的距离
    [14, 18, 21, 15, 22, 27, 0, 25, 30, 21],  # 城市 6 到 其他城市的距离
    [24, 22, 17, 12, 13, 26, 25, 0, 19, 21],  # 城市 7 到 其他城市的距离
    [21, 13, 15, 14, 17, 23, 30, 19, 0, 25],  # 城市 8 到 其他城市的距离
    [15, 18, 25, 17, 10, 20, 21, 21, 25, 0]   # 城市 9 到 其他城市的距离
])
# 初始信息素矩阵
pheromone_levels = np.ones((NUM_CITIES, NUM_CITIES)) * 0.1

# 启发式信息矩阵（距离的倒数）
heuristic_info = 1 / (distances + np.eye(NUM_CITIES))  # 避免除以0

# 定义蚂蚁类
class Ant:
    def __init__(self):
        self.path = []  # 蚂蚁走过的路径
        self.distance = 0  # 蚂蚁总行程

    def choose_next_city(self, current_city, pheromone_levels, heuristic_info):
        # 获取当前城市的所有可选择的城市
        allowed_cities = [i for i in range(NUM_CITIES) if i not in self.path]
        
        # 计算每条路径的选择概率
        probabilities = []
        total_prob = 0
        for city in allowed_cities:
            pheromone = pheromone_levels[current_city][city] ** ALPHA
            heuristic = heuristic_info[current_city][city] ** BETA
            prob = pheromone * heuristic
            probabilities.append(prob)
            total_prob += prob
        
        # 归一化概率
        probabilities = [prob / total_prob for prob in probabilities]

        # 使用轮盘赌选择下一城市
        next_city = random.choices(allowed_cities, probabilities)[0]
        return next_city

    def calculate_total_distance(self, distances):
        self.distance = 0
        for i in range(len(self.path) - 1):
            self.distance += distances[self.path[i], self.path[i + 1]]
        self.distance += distances[self.path[-1], self.path[0]]  # 回到起点
        return self.distance

# 蚁群算法主函数
def ant_colony_optimization():
    global pheromone_levels
    best_path = None
    best_distance = float('inf')

    for iteration in range(ITERATIONS):
        all_ants = [Ant() for _ in range(NUM_ANTS)]
        for ant in all_ants:
            # 从随机城市开始
            current_city = 0
            ant.path = [current_city]

            # 让蚂蚁选择路径直到走完所有城市
            while len(ant.path) < NUM_CITIES:
                next_city = ant.choose_next_city(current_city, pheromone_levels, heuristic_info)
                ant.path.append(next_city)
                current_city = next_city

            # 计算蚂蚁走过的总距离
            distance = ant.calculate_total_distance(distances)
            # 更新最短路径
            if distance < best_distance:
                best_distance = distance
                best_path = ant.path

        # 更新信息素
        pheromone_levels = pheromone_levels * (1 - RHO)  # 信息素挥发
        for ant in all_ants:
            # 对每条蚂蚁的路径增加信息素
            for i in range(len(ant.path) - 1):
                pheromone_levels[ant.path[i], ant.path[i + 1]] += Q / ant.distance
            pheromone_levels[ant.path[-1], ant.path[0]] += Q / ant.distance  # 最后一条路径回到起点

        # 打印当前迭代的最优路径和距离
        print(f"Iteration {iteration + 1}, Best Distance: {best_distance}, Path: {best_path}")
    
    return best_path, best_distance

# 执行蚁群算法
best_path, best_distance = ant_colony_optimization()
print(f"Best Path: {best_path}")
print(f"Best Distance: {best_distance}")
