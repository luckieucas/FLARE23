import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import time


def get_bounding_box(x, margins):
    """Calculates the bounding box of a ndarray"""
    mask = x == 0
    bbox = []
    all_axis = np.arange(x.ndim)

    for kdim in all_axis:
        nk_dim = np.delete(all_axis, kdim)
        mask_i = mask.all(axis=tuple(nk_dim))
        dmask_i = np.diff(mask_i)
        idx_i = np.nonzero(dmask_i)[0]
        # idx_i = idx_i[0]
        # if len(idx_i) != 2:
        #     raise ValueError(
        #         "Algorithm failed, {} does not have 2 elements!".format(idx_i)
        #     )
        if len(idx_i) != 0:
            start = max(0, idx_i[0] + 1 - margins[kdim])
            end = min(x.shape[kdim], idx_i[-1] + 1 + margins[kdim])
        else:
            start = 0
            end = x.shape[kdim]

        bbox.append(slice(start, end))
    return tuple(bbox)


def RemoveSmallStructures(label_np, exclude_list, threshold, voxel_volume, min_volumes):
    for l in np.arange(1, 14):
        if (l in label_np) and (l not in exclude_list):
            volume = np.count_nonzero(label_np == l) * voxel_volume
            if volume < (threshold * min_volumes[l]):
                label_np[label_np == l] = 0
    return label_np


def CloseStructures(label_np, structures):
    for s in structures:
        if s in label_np:
            label_s_bin = np.zeros(label_np.shape)
            label_s_bin[label_np == s] = 1
            bbox = get_bounding_box(label_s_bin, [5, 5, 5])
            croped_label = label_s_bin[bbox]
            closed_s = ndimage.binary_closing(
                croped_label, structure=np.ones((5, 5, 5))
            ).astype(np.int8)
            label_s_bin[bbox] = closed_s
            label_np[label_s_bin == 1] = s

    return label_np


def RemoveConnectedComponents(label_np, threshold, voxel_volume, exclude_list):
    for i in np.arange(1, 14):
        if (i in label_np) and (i not in exclude_list):
            label_s = np.zeros(np.shape(label_np))
            label_s[label_np == i] = 1
            labeled_array, num_features = ndimage.label(
                label_s, structure=np.ones((3, 3, 3))
            )
            if num_features > 1:
                whole_volume = np.count_nonzero(label_s == 1) * voxel_volume
                for j in range(num_features + 1):
                    if (np.count_nonzero(labeled_array == j) * voxel_volume) < (
                        threshold * whole_volume
                    ):
                        labeled_array[labeled_array == j] = 0
                label_np[label_np == i] = 0
                label_np[labeled_array != 0] = i
    return label_np


def FixGallblader(label_np):
    label_gb = np.copy(label_np)
    label_gb[label_gb != 9] = 0
    label_gb[label_gb == 9] = 1
    labeled_array, num_features = ndimage.label(label_gb, structure=np.ones((3, 3, 3)))
    if num_features > 1:
        enclosed = []
        for f in range(1, num_features + 1):
            labeled_array_ccp = np.zeros(labeled_array.shape)
            labeled_array_ccp[labeled_array == f] = 1
            for i in range(labeled_array_ccp.shape[0]):
                current_slice = labeled_array_ccp[i, :, :]
                if np.any(current_slice):
                    dilated_image = ndimage.binary_dilation(current_slice).astype(
                        np.int8
                    )
                    countour_ccp = dilated_image - current_slice
                    contour_original = np.copy(label_np[i, :, :])
                    contour_original[countour_ccp != 1] = 0
                    if set(contour_original[countour_ccp == 1].flatten()) == {1}:
                        enclosed.append(f)
                        label_np[labeled_array == f] = 1
                        break
    return label_np


def MaskWithVessels(label_np, z_size):
    start = 0
    stop = 0
    slices_gap = int(20 / z_size)
    structs = {5, 6, 10}
    for i in reversed(range(label_np.shape[0])):
        set_slice = set(label_np[i, :, :].flatten())
        if start == 0 and len(set_slice & structs) > 0:
            start = i
        if start != 0 and stop == 0 and len(set_slice & structs) == 0:
            stop = i
            check_slice = max(0, stop - slices_gap)
            if len(set(label_np[check_slice, :, :].flatten()) & structs) != 0:
                stop = 0
        if start != 0 and stop != 0:
            break

    slices_gap = int(20 / z_size)
    stop_slice = max(0, stop - slices_gap)
    start_slice = min(start, label_np.shape[0])

    label_np[:stop_slice] = 0
    label_np[start_slice:] = 0

    return label_np


def postprocess_file(filepath):
    """This function should:

    1. Load nifti from filepath
    2. Do stuff with the mask
    3. Overwrite the nifti in filepath

    filepath will be a pathlib Path, so maybe you need to str(filepath) for sitk
    """

    min_volumes = {
        1: 1037420.7294452335,
        2: 107952.71351374676,
        3: 51560.09455919893,
        4: 41522.03935737655,
        5: 40533.60780851278,
        6: 41437.472288146426,
        7: 1385.5876138624267,
        8: 2445.168267657047,
        9: 4276.847870821996,
        10: 6978.699080946171,
        11: 140339.32771712207,
        12: 43614.98921844564,
        13: 102572.24069653488,
    }

    label_sitk = sitk.ReadImage(str(filepath))
    label_np = sitk.GetArrayFromImage(label_sitk)

    tic = time.time()
    # Exclude small structures
    label_spacing = label_sitk.GetSpacing()
    voxel_volume = label_spacing[0] * label_spacing[1] * label_spacing[2]
    excluded_organs = [1, 5, 6, 10, 12, 11]
    label_remove_small = RemoveSmallStructures(
        label_np, excluded_organs, 0.5, voxel_volume, min_volumes
    )
    toc = time.time()
    print(f"Excluding small structures took {toc-tic}")

    tic = time.time()
    # Close aorta, IVC, RAG, LAG
    structures_to_close = [7, 8]
    closed_label_np = CloseStructures(label_remove_small, structures_to_close)
    toc = time.time()
    print(f"Closing structures took {toc-tic}")

    tic = time.time()
    # Masking with vessels
    masked_label_np = MaskWithVessels(closed_label_np, label_spacing[2])
    toc = time.time()
    print(f"Masking with vessels took {toc-tic}")

    tic = time.time()
    # Remove connected components that are less than 5% total volume
    exclude_list_cca = [9]
    removed_components_label_np = RemoveConnectedComponents(
        masked_label_np, 0.1, voxel_volume, exclude_list_cca
    )
    toc = time.time()
    print(f"Removing connected components took {toc-tic}")

    tic = time.time()
    # Fix gallbladder
    fix_gallblader = FixGallblader(removed_components_label_np)
    toc = time.time()
    print(f"Fixing gallbladder took {toc-tic}")

    new_label = sitk.GetImageFromArray(fix_gallblader)
    new_label.CopyInformation(label_sitk)
    sitk.WriteImage(new_label, str(filepath))