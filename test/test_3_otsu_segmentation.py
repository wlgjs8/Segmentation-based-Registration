import os
import numpy as np
import json
import nibabel as nib
from scipy import ndimage
from skimage import filters
from skimage.measure import label


from config import (
    RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH, UPPER_TOOTH_NUM, LOWER_TOOTH_NUM, OTSU_PERSON_INDEX
)


def draw_whole_box(box_volume, ind, label='metal'):
    # print('ind : ', ind)
    # print('shape : ', shape)

    if label == 'metal':
        color = 2
    else:
        color = 1

    # 두 점으로 정의된 박스의 꼭지점 좌표
    point1 = (ind[0], ind[1], ind[2])
    point2 = (ind[3], ind[4], ind[5])

    # 박스의 시작 지점과 크기 계산
    min_x = min(point1[0], point2[0])
    min_y = min(point1[1], point2[1])
    min_z = min(point1[2], point2[2])
    max_x = max(point1[0], point2[0])
    max_y = max(point1[1], point2[1])
    max_z = max(point1[2], point2[2])

    # 면 영역만을 color로 채우기 위한 복사본 생성
    box_volume_copy = np.copy(box_volume)

    # 면 영역 가장자리를 color로 채우기
    box_volume_copy[min_x:max_x+1, min_y:max_y+1, min_z] = color
    box_volume_copy[min_x:max_x+1, min_y:max_y+1, max_z] = color
    box_volume_copy[min_x:max_x+1, min_y, min_z:max_z+1] = color
    box_volume_copy[min_x:max_x+1, max_y, min_z:max_z+1] = color
    box_volume_copy[min_x, min_y:max_y+1, min_z:max_z+1] = color
    box_volume_copy[max_x, min_y:max_y+1, min_z:max_z+1] = color

    # 원래 볼륨에 복사
    box_volume = np.copy(box_volume_copy)

    return box_volume


def half_fill_whole_box(box_volume, ind, label='metal'):

    # print('shape : ', shape)

    if label == 'metal':
        color = 0
    else:
        color = 1

    
    point1 = (ind[0], ind[1], ind[2])
    # point2 = (ind[3], ind[4], ind[5])
    point2 = ((ind[0] + ind[3])//2, ind[4], ind[5])

    # 박스의 크기 계산
    box_size = [abs(point1[0] - point2[0]) + 1,
                abs(point1[1] - point2[1]) + 1,
                abs(point1[2] - point2[2]) + 1]

    # 박스의 시작 지점 계산
    start_point = [min(point1[0], point2[0]),
                min(point1[1], point2[1]),
                min(point1[2], point2[2])]
    
    # box_size[0] = box_size[0]//2
    # start_point[0] = start_point[0] + box_size[0]

    # 색상으로 채우기 위한 박스 생성
    color_box = np.full(box_size, color)

    # 박스를 원래 볼륨에 복사
    box_volume[start_point[0]:start_point[0]+box_size[0],
            start_point[1]:start_point[1]+box_size[1],
            start_point[2]:start_point[2]+box_size[2]] = color_box
    
    return box_volume


def add_margin(minx, miny, minz, maxx, maxy, maxz, MARGIN=3.5):
    minx -= MARGIN
    miny -= MARGIN
    minz -= MARGIN

    maxx += MARGIN
    maxy += MARGIN
    maxz += MARGIN

    if minx < 0:
        minx = 0
    if miny < 0:
        miny = 0
    if minz < 0:
        minz = 0

    if maxx >= RESIZE_DEPTH:
        maxx = RESIZE_DEPTH -1
    if maxy >= RESIZE_HEIGHT:
        maxy = RESIZE_HEIGHT -1
    if maxz >= RESIZE_WIDTH:
        maxz = RESIZE_WIDTH -1

    return minx, miny, minz, maxx, maxy, maxz

    
def _get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1) -> float:
    # return min(2 * matrix.mean(), max_value)
    return 2*matrix.mean()

def otsu_thresholding(filled_volume, image):
    volume_data = filled_volume * image

    otsu_threshold = filters.threshold_otsu(volume_data)
    print('image otsu_threshold : ', otsu_threshold)
    binary_volume = volume_data > otsu_threshold

    return binary_volume.astype(int)

def single_otsu_thresholding(filled_volume, original_image):
    volume_data = filled_volume * original_image
    single_otsu_threshold = filters.threshold_otsu(volume_data)
    print('single_otsu_threshold : ', single_otsu_threshold)

    binary_volume = np.zeros(original_image.shape)
    binary_volume[filled_volume==1] = volume_data[filled_volume==1] > single_otsu_threshold

    return binary_volume.astype(int), single_otsu_threshold


def ccl_whole_box(ccl_volume, original_image, ind, threshold=1717.1131525039673):
    
    point1 = (ind[0], ind[1], ind[2])
    point2 = ((ind[0] + ind[3])//2, ind[4], ind[5])
    box_size = [abs(point1[0] - point2[0]) + 1,
                abs(point1[1] - point2[1]) + 1,
                abs(point1[2] - point2[2]) + 1]
    start_point = [min(point1[0], point2[0]),
                min(point1[1], point2[1]),
                min(point1[2], point2[2])]

    target_volume = original_image[start_point[0]:start_point[0]+box_size[0],
            start_point[1]:start_point[1]+box_size[1],
            start_point[2]:start_point[2]+box_size[2]]
    # print('target_volume : ', target_volume)
    target_otsu_result = target_volume > threshold
    target_otsu_result = target_otsu_result.astype(int)

    labeled_array, num_features = ndimage.label(target_otsu_result)

    unique_labels, label_counts = np.unique(labeled_array, return_counts=True)
    largest_object_label = unique_labels[np.argmax(label_counts[1:]) + 1]
    largest_component_mask = (labeled_array == largest_object_label)
    largest_connected_component = target_otsu_result * largest_component_mask

    ccl_volume[start_point[0]:start_point[0]+box_size[0],
            start_point[1]:start_point[1]+box_size[1],
            start_point[2]:start_point[2]+box_size[2]] = largest_connected_component

    return ccl_volume


DATA_DIR = os.path.abspath("C:/Users/JeeheonKim/source/ct/pose/ct_model_box")
SAVE_DIR = os.path.abspath("C:/Users/JeeheonKim/source/ct/pose/ct_model_box")

datalist = []
for i in range(len(OTSU_PERSON_INDEX)):
    datalist.append((OTSU_PERSON_INDEX[i], 'upper'))
    datalist.append((OTSU_PERSON_INDEX[i], 'lower'))


for i in range(len(datalist)):
    person_id, flag = datalist[i]
    print('Idx : {}, Person ID : {}, flag : {}'.format(i, person_id, flag))

    if flag == 'upper':
        img_name = 'crop_image_upper.nii.gz'
        # box_anno = 'upper_bbox.json'
        # box_anno = 'box_pred_upper_all.json'
        box_anno = 'box_pred_upper_no.json'
        TOOTH_NUM = UPPER_TOOTH_NUM
    elif flag == 'lower':
        img_name = 'crop_image.nii.gz'
        # box_anno = 'lower_bbox.json'
        # box_anno = 'box_pred_lower_all.json'
        box_anno = 'box_pred_lower_no.json'
        TOOTH_NUM = LOWER_TOOTH_NUM

    img_path = os.path.join(DATA_DIR, person_id, img_name)
    img_object = nib.load(img_path)
    original_image = img_object.get_fdata()
    original_shape = original_image.shape

    box_volume = np.zeros(original_shape)
    fill_volume = np.zeros(original_shape)
    otsu_concat = np.zeros(original_shape)
    ccl_concat = np.zeros(original_shape)

    with open(os.path.join(DATA_DIR, person_id, box_anno), 'r') as file:
        bbox_annos = json.load(file)
    
    for TOOTH_IDX in TOOTH_NUM:
        if TOOTH_IDX not in bbox_annos.keys():
            continue

        print('TOOTH_IDX : ', TOOTH_IDX)

        minx, miny, minz, maxx, maxy, maxz = bbox_annos[TOOTH_IDX]
        minx, miny, minz, maxx, maxy, maxz = add_margin(minx, miny, minz, maxx, maxy, maxz)

        minx = int(minx * 1.75)
        miny = int(miny * 1.75)
        minz = int(minz * 1.75)
        maxx = int(maxx * 1.75)
        maxy = int(maxy * 1.75)
        maxz = int(maxz * 1.75)

        label = 'normal'
        box_volume = draw_whole_box(box_volume, (minx, miny, minz, maxx, maxy, maxz), label=label)
        fill_volume = half_fill_whole_box(fill_volume, (minx, miny, minz, maxx, maxy, maxz), label=label)

        # single_fill_volume = np.zeros((maxx-minx, maxy-miny, maxz-minz))
        single_fill_volume = np.zeros(original_shape)
        single_fill_volume = half_fill_whole_box(single_fill_volume, (minx, miny, minz, maxx, maxy, maxz), label=label)
        # print('single_fill_volume : ', single_fill_volume.shape)
        single_otsu_volume, single_otsu_threshold = single_otsu_thresholding(single_fill_volume, original_image)
        single_ccl_volume = np.zeros(original_shape)
        single_ccl_volume = ccl_whole_box(single_ccl_volume, original_image, (minx, miny, minz, maxx, maxy, maxz), threshold=single_otsu_threshold)
        ccl_concat = np.logical_or(ccl_concat, single_ccl_volume) * 1

        otsu_concat = np.logical_or(otsu_concat, single_otsu_volume) * 1
        # otsu_concat = (otsu_concat | single_otsu_volume) * 1
        # save_otsu_path = os.path.join(SAVE_DIR, person_id, 'single_otsu_{}_result_ttt{}.nii.gz'.format(flag, TOOTH_IDX))
        # nii_otsu_volume = nib.Nifti1Image(single_fill_volume, affine=np.eye(4))
        # nib.save(nii_otsu_volume, save_otsu_path)

    # save_otsu_path = os.path.join(SAVE_DIR, person_id, 'single_otsu_{}_result_concat.nii.gz'.format(flag))
    # nii_otsu_volume = nib.Nifti1Image(otsu_concat, affine=np.eye(4))
    # nib.save(nii_otsu_volume, save_otsu_path)

    save_ccl_path = os.path.join(SAVE_DIR, person_id, 'otsu_{}_result.nii.gz'.format(flag))
    nii_ccl_volume = nib.Nifti1Image(ccl_concat, affine=np.eye(4))
    nib.save(nii_ccl_volume, save_ccl_path)