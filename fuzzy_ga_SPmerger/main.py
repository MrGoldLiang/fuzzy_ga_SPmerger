from skimage.segmentation import slic, find_boundaries
from skimage import io,color
import skfuzzy as fuzz
import numpy as np
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries


from scipy import stats


def get_superpixel_neighbors(segments):

    neighbors = {}
    for label in np.unique(segments):
        mask = segments == label
        boundaries = find_boundaries(mask, mode='outer')
        neighbor_labels = np.unique(segments[boundaries])
        nlist = []
        for neighbor_label in neighbor_labels:
            if neighbor_label != label:
                nlist.append(neighbor_label)
        neighbors[label] = nlist

    return neighbors


def normalize(data_list):
    # 找出列表中的最小值和最大值
    # 将 data_list 转换为形状为 (n,3) 的 NumPy 数组
    data_array = np.vstack(data_list)

    # 对每一列进行最大最小归一化
    data_norm = (data_array - np.min(data_array, axis=0)) / (np.max(data_array, axis=0) - np.min(data_array, axis=0))

    data_norm_list = [np.array(data) for data in data_norm]
    return data_norm_list


def get_superpixel_features(image, segments):
    neighbors = get_superpixel_neighbors(segments)
    textures = []
    colors = []

    for i in np.unique(segments):
        mask = segments == i
        texture = image[mask].std(axis=0)

        color = np.mean(image[mask], axis=0)
        textures.append(texture)
        colors.append(color)
    # print(len(textures))
    return normalize(textures), normalize(colors), neighbors

def ctrl_get(vars):

    texture_sim = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'texture_sim')
    color_sim = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'color_sim')
    merge = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'merge')

    color_sim['poor'] = fuzz.gaussmf(x=color_sim.universe, mean=vars[6], sigma=vars[7])
    color_sim['average'] = fuzz.gaussmf(x=color_sim.universe, mean=vars[8], sigma=vars[9])
    color_sim['good'] = fuzz.gaussmf(x=color_sim.universe, mean=vars[10], sigma=vars[11])

    texture_sim['poor'] = fuzz.gaussmf(texture_sim.universe, sigma=vars[0], mean=vars[1])
    texture_sim['average'] = fuzz.gaussmf(texture_sim.universe, sigma=vars[2], mean=vars[3])
    texture_sim['good'] = fuzz.gaussmf(texture_sim.universe, sigma=vars[4], mean=vars[5])

    # texture_sim.automf(3)
    # color_sim.automf(3)

    texture_sim.view()
    color_sim.view()
    plt.show()
    merge.automf(3)

    rule1 = ctrl.Rule(texture_sim['poor']|color_sim['poor'],consequent=merge['poor'])

    rule2 = ctrl.Rule(texture_sim['average']&color_sim['average'],consequent=merge['average'])

    rule3 = ctrl.Rule(color_sim['good'],consequent=merge['good'])

    merge_ctrl = ctrl.ControlSystem([rule1,rule2,rule3])
    merging = ctrl.ControlSystemSimulation(merge_ctrl)

    return merging

def plot_array_distribution(arr_list):
    # 将列表转换为二维数组
    arr = np.array(arr_list)

    # 获取每列数据的分布情况
    col1 = arr[:, 0]
    col2 = arr[:, 1]
    col3 = arr[:, 2]
    # 绘制每列数据的频率直方图
    plt.hist(col1, alpha=0.5, label='Red',bins=20)
    plt.hist(col2, alpha=0.5, label='Green',bins=20)
    plt.hist(col3, alpha=0.5, label='Blue',bins=20)
    plt.legend(loc='upper right')
    plt.show()


def merge_superpixels(superpixel_dict, segmentation):
    visited = set()
    def dfs(label):
        visited.add(label)
        # Get the list of superpixel labels to be merged into the current label.
        merge_labels = superpixel_dict.get(label, [])
        # Merge the connected superpixels with the same label.
        for merge_label in merge_labels:
            # Skip the visited labels to avoid infinite loop.
            if merge_label not in visited:
                # Recursively merge the connected superpixels.
                dfs(merge_label)
            # Merge the current superpixel label with the merged superpixel label.
            segmentation[segmentation == merge_label] = label

    # Merge the superpixels according to the dictionary.
    for label in superpixel_dict:
        # Skip the visited labels to avoid duplicate merging.
        if label not in visited:
            dfs(label)

    return segmentation






if __name__ == '__main__':
    image = io.imread('4_3.tif')
    segments = slic(image,n_segments= 500)

    textures, colors, neighbors = get_superpixel_features(image,segments)

    # plot_array_distribution(textures)
    # plot_array_distribution(colors)

    vars = [0.10389423459971904,0.13864804175560952,9.636743068695068e-07,0.9958200458891868,0.8178711119628906,0.5559387251437379,0.1940631947048759,0.22995002817687993,0.4427376350719452,1e-07,1.0,0.19433601806640627]
    merger = ctrl_get(vars)
    remerge_dict = {}
    merge_l = []
    for key,value in neighbors.items():
        merge_list = {}
        for neighbor in value:
            texture_sim = 1 - np.linalg.norm(textures[key-1] - textures[neighbor-1])
            color_sim = 1 - np.linalg.norm(colors[key-1] - colors[neighbor-1])

            merger.input['texture_sim'] = texture_sim
            merger.input['color_sim'] = color_sim
            merger.compute()
            # if merger.output['merge'] > 0.85:
            merge_l.append(merger.output['merge'])
            merge_list[neighbor] = merger.output['merge']
        remerge_dict[key] = merge_list

    mean = sum(merge_l) / len(merge_l)
    # std = np.std(merge_l)
    # norm_lst = stats.norm(mean, std).cdf(merge_l)



    merge_dict = {}



    for key,item in remerge_dict.items():
        merge_list = []
        for neighbor,value in item.items():
            if value/mean > 1.8:
                merge_list.append(neighbor)
        merge_dict[key] = merge_list

    original_image = mark_boundaries(image,segments)
    segments = merge_superpixels(merge_dict,segments)
    segmented_image = mark_boundaries(image, segments)
    io.imsave('original_image.jpg',original_image)
    io.imsave('merge_image.jpg', segmented_image)






