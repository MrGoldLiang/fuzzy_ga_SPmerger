from skfuzzy import control as ctrl
import numpy as np
import geatpy as ea
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import main as fm
from random import sample
from PIL import Image
from skimage.segmentation import mark_boundaries

merge_texture_sim_list = []
merge_color_sim_list = []
unmerge_texture_sim_list = []
unmerge_color_sim_list = []



def mark_superpixel(image, segmentation, label, border_color=(255, 0, 0), thickness=1):
    # 将输入图像和分割结果转换为numpy数组类型
    if isinstance(image, Image.Image):
        image = np.array(image)
    if isinstance(segmentation, Image.Image):
        segmentation = np.array(segmentation)

    # 使用mark_boundaries函数绘制超像素边界
    boundaries = mark_boundaries(image, segmentation, color=border_color)

    # 绘制目标超像素边界
    mask = segmentation == label
    mask_indices = np.where(mask)
    for i in range(thickness):
        boundaries[mask_indices[0] - i, mask_indices[1], :] = border_color
        boundaries[mask_indices[0] + i, mask_indices[1], :] = border_color
        boundaries[mask_indices[0], mask_indices[1] - i, :] = border_color
        boundaries[mask_indices[0], mask_indices[1] + i, :] = border_color

    # 将numpy数组转换为PIL图像对象
    overlaid_image = np.maximum(image, boundaries)

    # 将numpy数组转换为PIL图像对象
    pil_image = Image.fromarray(np.uint8(overlaid_image))

    return pil_image







image = fm.io.imread('512_2.tif')

segments = fm.slic(image,n_segments= 500)

textures, colors, neighbors = fm.get_superpixel_features(image,segments)


# 随机采样
professor = sample(neighbors.keys(),10)

# 专家采样
# professor = []
# def onclick(event):
#     if event.xdata is not None and event.ydata is not None:
#         x = int(round(event.xdata))
#         y = int(round(event.ydata))
#         label = segments[y, x]
#         print(f"Clicked on label {label}")
#         professor.append(label)
#         if len(professor) == 5:
#             fig.canvas.mpl_disconnect(cid)  # 断开事件连接，停止交互模式
# fig, ax = plt.subplots()
# ax.imshow(image)
# boundaries = mark_boundaries(image, segments)
# ax.imshow(boundaries)
# ax.set_axis_off()
# cid = fig.canvas.mpl_connect('button_press_event', onclick)
# plt.show()



sample_dict = {}
for item in professor:
    neighbors_list = neighbors[item]
    self_texture = textures[item-1]
    self_color = colors[item-1]
    neighbor_sim = {}
    for neighbor in neighbors_list:
        neighbor_texture = textures[neighbor-1]
        neighbor_color = colors[neighbor-1]
        texture_sim = 1 - np.linalg.norm(self_texture-neighbor_texture)
        color_sim = 1 - np.linalg.norm(self_color-neighbor_color)
        neighbor_sim[neighbor] = [texture_sim,color_sim]
    sample_dict[item] = neighbor_sim

# print(sample_dict)
professor_mark =  {}
for key,neighbor_list in sample_dict.items():
    print('curneighbours = ',format(neighbor_list))
    show_label = mark_superpixel(image,segments,key)
    plt.imshow(show_label)
    plt.show()
    mark_dict = {}
    for n in neighbor_list.keys():
        print(n)
        show_label = mark_superpixel(image, segments, n,border_color=(0,255,0))
        plt.imshow(show_label)
        plt.show()
        mark = input()
        mark_dict[n] = mark
    professor_mark[key] = mark_dict

# print(professor_mark)

for key,mark_list in professor_mark.items():
    for n,mark in mark_list.items():
        feature = sample_dict[key][n]
        t_s = feature[0]
        c_s = feature[1]
        if mark == '1':
            merge_texture_sim_list.append(t_s)
            merge_color_sim_list.append(c_s)
        if mark == '0':
            unmerge_texture_sim_list.append(t_s)
            unmerge_color_sim_list.append(c_s)




def ctrl_text(vars):
    global merge_texture_sim_list, merge_color_sim_list, unmerge_texture_sim_list, unmerge_color_sim_list
    texture_sim = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'texture_sim')
    color_sim = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'color_sim')
    merge = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'merge')

    color_sim['poor'] = fuzz.gaussmf(x=color_sim.universe, mean=vars[6], sigma=vars[7])
    color_sim['average'] = fuzz.gaussmf(x=color_sim.universe, mean=vars[8], sigma=vars[9])
    color_sim['good'] = fuzz.gaussmf(x=color_sim.universe, mean=vars[10], sigma=vars[11])

    texture_sim['poor'] = fuzz.gaussmf(texture_sim.universe, sigma=vars[0], mean=vars[1])
    texture_sim['average'] = fuzz.gaussmf(texture_sim.universe, sigma=vars[2], mean=vars[3])
    texture_sim['good'] = fuzz.gaussmf(texture_sim.universe, sigma=vars[4], mean=vars[5])

    merge.automf(3)

    rule1 = ctrl.Rule(texture_sim['poor'] | color_sim['poor'], consequent=merge['poor'])
    rule2 = ctrl.Rule(texture_sim['average'] & color_sim['average'], consequent=merge['average'])
    rule3 = ctrl.Rule(color_sim['good'], consequent=merge['good'])

    merge_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    merging = ctrl.ControlSystemSimulation(merge_ctrl)
    merge_list = []
    unmerge_list = []
    for t_sim, c_sim in zip(merge_texture_sim_list, merge_color_sim_list):
        merging.input['texture_sim'] = t_sim
        merging.input['color_sim'] = c_sim
        merging.compute()
        merge_result = merging.output['merge']



        merge_list.append(merge_result)
    for t_sim, c_sim in zip(unmerge_texture_sim_list, unmerge_color_sim_list):
        merging.input['texture_sim'] = t_sim
        merging.input['color_sim'] = c_sim
        merging.compute()
        merge_result = merging.output['merge']


        unmerge_list.append(merge_result)

    if len(merge_list) > 0:
        merge_mean = sum(merge_list) / len(merge_list)
    else:
        merge_mean = 0
    if len(unmerge_list) > 0:
        unmerge_mean = sum(unmerge_list) / len(unmerge_list)
    else:
        unmerge_mean = 0

    return merge_mean, unmerge_mean


# 构建问题
r = 1  # 目标函数需要用到的额外数据


@ea.Problem.single
def evalVars(Vars):  # 定义目标函数（含约束）
    f1, f2 = ctrl_text(Vars)
    f = f1 - f2  # 计算目标函数值
    # print(f)
    # 没想好约束函数是啥

    # CV = np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9])  # 计算违反约束程度
    return f


problem = ea.Problem(name='soea quick start demo',
                     M=1,  # 目标维数
                     maxormins=[-1],  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
                     Dim=12,  # 决策变量维数
                     varTypes=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 决策变量的类型列表，0：实数；1：整数
                     lb=[0.000000001, 0.00000001, 0.00000001, 0.0000001,0.0000001, 0.00000001, 0.00000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001,],  # 决策变量下界
                     ub=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 决策变量上界
                     evalVars=evalVars)
# 构建算法
algorithm = ea.soea_SEGA_templet(problem,
                                 ea.Population(Encoding='RI', NIND=20),
                                 MAXGEN=1000,  # 最大进化代数。
                                 logTras=2,  # 表示每隔多少代记录一次日志信息，0表示不记录。
                                 trappedValue=1e-8,  # 单目标优化陷入停滞的判断阈值。
                                 maxTrappedCount=10)  # 进化停滞计数器最大上限值。
# 求解

res = ea.optimize(algorithm, seed=1, verbose=True, drawing=1, outputMsg=True, drawLog=False, saveFlag=True,
                  dirName='result')
