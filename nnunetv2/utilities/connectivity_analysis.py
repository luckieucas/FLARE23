import numpy as np
from scipy import ndimage


def connectivity_analyze(seg_result, min_area = 100):
    labels = np.unique(seg_result)
    connected_regions = {}
    for label in labels:
        binary_mask = seg_result == label
        labeled_mask, num_regions = ndimage.label(binary_mask)
        connected_regions[label] = ndimage.find_objects(labeled_mask)
       # print(label, labeled_mask, connected_regions[label])
    
   # min_area = 100
    for label in labels:
        regions = connected_regions[label]
        valid_regions = []
        for region in regions:
            if region is None:
                continue
            height = region[0].stop - region[0].start
            width = region[1].stop - region[1].start
            area = height * width
            if area >= min_area:
                valid_regions.append(region)
        connected_regions[label] = valid_regions

    pixel_labels = np.zeros_like(seg_result)
    for label, regions in connected_regions.items():
        for region in regions:
            binary_mask = np.zeros_like(seg_result, dtype=np.bool)
            binary_mask[region] = True
            region_pixels = seg_result[binary_mask]
            region_labels, counts = np.unique(region_pixels, return_counts=True)
            max_label = region_labels[np.argmax(counts)]
            pixel_labels[binary_mask] = max_label

    merged = np.zeros_like(seg_result)
    for label in labels:
        binary_mask = pixel_labels == label
        merged[binary_mask] = label

    return merged


#a = np.arange(120).reshape(2,3,4,5)
#print(connectivity_analyze(a))


### 假设有一个多类别分割结果，每个像素值表示对应的类别
##seg_result = np.array([
##    [0, 0, 1, 1],
##    [0, 2, 2, 1],
##    [3, 2, 2, 1],
##    [3, 3, 3, 3]
##], dtype=np.int)
##
### 对每个类别进行连通域分割
##labels = np.unique(seg_result)
##connected_regions = {}
##for label in labels:
##    binary_mask = seg_result == label
##    labeled_mask, num_regions = ndimage.label(binary_mask)
##    connected_regions[label] = ndimage.find_objects(labeled_mask)
##    print(label, labeled_mask, connected_regions[label])
##
### 对于每个类别，去除面积过小的连通域
##min_area = 3
##for label in labels:
##    regions = connected_regions[label]
##    valid_regions = []
##    for region in regions:
##        if region is None:
##            continue
##        height = region[0].stop - region[0].start
##        width = region[1].stop - region[1].start
##        area = height * width
##        if area >= min_area:
##            valid_regions.append(region)
##    connected_regions[label] = valid_regions
##
### 将每个像素分配到最大的连通域所属的类别中
##pixel_labels = np.zeros_like(seg_result)
##for label, regions in connected_regions.items():
##    for region in regions:
##        binary_mask = np.zeros_like(seg_result, dtype=np.bool)
##        binary_mask[region] = True
##        region_pixels = seg_result[binary_mask]
##        region_labels, counts = np.unique(region_pixels, return_counts=True)
##        max_label = region_labels[np.argmax(counts)]
##        pixel_labels[binary_mask] = max_label
##
### 将所有类别的分割结果合并到一个分割结果中
##merged = np.zeros_like(seg_result)
##for label in labels:
##    binary_mask = pixel_labels == label
##    merged[binary_mask] = label
##
### 输出结果
##print("原始分割结果：")
##print(seg_result)
##print("去除噪声后的分割结果：")
##print(merged)
