# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import annotations

import logging
import os
import json
import copy
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pycocotools.coco import COCO


logger = logging.getLogger(__name__)


colors = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'gray': (128, 128, 128),
    'white': (255, 255, 255),
    'black': (0, 0, 0)}


def readTXT(txt_path):
    with open(txt_path, 'r') as f:
        listInTXT = [line.strip() for line in f]

    return listInTXT


class PoseDataset(Dataset):
    def __init__(self, root, image_set, is_train, max_prompt_num=5, min_prompt_num=1,
        radius=10, size=256, transparency=0.0, sample_weight=1.0, transform=None):
        
        self.sample_weight = sample_weight
        self.max_prompt_num = max_prompt_num
        self.min_prompt_num = min_prompt_num
        self.radius = radius
        self.transparency = transparency
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []
        
        self.keypoints_type = {}
        
        self.is_train = is_train
        self.image_set = image_set
        self.root = root

        self.scale_factor = 0.35
        self.rotation_factor = 45
        self.flip = True
        self.num_joints_half_body = 8
        self.prob_half_body = 0.3

        self.image_size = np.array((size, size))
        self.heatmap_size = np.array((size, size))

        self.transform = transform
        self.db = []

        pose_diverse_prompt_path = 'dataset/prompt/prompt_pose.txt'
        self.pose_diverse_prompt_list = []
        with open(pose_diverse_prompt_path) as f:
            line = f.readline()
            while line:
                line = line.strip('\n')
                self.pose_diverse_prompt_list.append(line)
                line = f.readline()

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self,):
        return int(len(self.db) * self.sample_weight)

    def __getitem__(self, idx):
        if self.sample_weight >= 1:
            idx = idx % len(self.db)
        else:
            idx = int(idx / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1)

        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, prompt = self.generate_target(input, joints, joints_vis)
        
        # return Image.fromarray(input), Image.fromarray(target), prompt

        image_0 = rearrange(2 * torch.tensor(np.array(input)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(target)).float() / 255 - 1, "h w c -> c h w")

        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))

    def generate_target(self, input, joints, joints_vis):
        '''
        :param input: [height, width, 3]
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target
        '''
        radius = self.radius
        target = copy.deepcopy(input)

        joint_num = random.randint(self.min_prompt_num, self.max_prompt_num)
        joint_ids = np.random.choice([i for i in range(self.num_joints)], joint_num, replace=False)
        random_color_names = random.sample(list(colors.keys()), len(joint_ids))
        random_marker_names = ['circle' for i in range(len(joint_ids))]

        prompt = ""

        for color_idx, joint_id in enumerate(joint_ids):
            feat_stride = self.image_size / self.heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - radius), int(mu_y - radius)]
            br = [int(mu_x + radius + 1), int(mu_y + radius + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                joints_vis[joint_id][0] = 0
                continue

            marker_size = 2 * radius + 1
            g = np.zeros((marker_size, marker_size))
            x, y = np.indices((marker_size, marker_size))
            interval = int((marker_size - marker_size / math.sqrt(2)) // 2)
            mask = (x - radius) ** 2 + (y - radius) ** 2 <= radius ** 2 + 1
            g[mask] = 1

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = joints_vis[joint_id][0]
            random_color_name = random_color_names[color_idx]
            random_color = colors[random_color_name]
            
            prompt += random.choice(self.pose_diverse_prompt_list).format(
                color=random_color_name, 
                joint=self.keypoints_type[joint_id])

            if v > 0.5:
                target[img_y[0]:img_y[1], img_x[0]:img_x[1]][g[g_y[0]:g_y[1], g_x[0]:g_x[1]]>0] \
                    = self.transparency*target[img_y[0]:img_y[1], img_x[0]:img_x[1]][g[g_y[0]:g_y[1], g_x[0]:g_x[1]]>0] \
                        + (1-self.transparency)*np.array(random_color)

        return target, prompt


class COCODataset(PoseDataset):
    def __init__(self, root, image_set, is_train, max_prompt_num=5, min_prompt_num=1, 
            radius=10, size=256, transparency=0.0, sample_weight=1.0, transform=None):

        super().__init__(root, image_set, is_train, max_prompt_num, min_prompt_num, 
            radius, size, transparency, sample_weight, transform)

        self.keypoints_type = {
                0: "nose",
                1: "left eye",
                2: "right eye",
                3: "left ear",
                4: "right ear",
                5: "left shoulder",
                6: "right shoulder",
                7: "left elbow",
                8: "right elbow",
                9: "left wrist",
                10: "right wrist",
                11: "left hip",
                12: "right hip",
                13: "left knee",
                14: "right knee",
                15: "left ankle",
                16: "right ankle"
            }

        self.image_width = size
        self.image_height = size
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200

        self.coco = COCO(self._get_ann_file_keypoint())

        # deal with class names
        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        logger.info('=> num_images: {}'.format(self.num_images))

        self.num_joints = 17
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        self.parent_ids = None
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)
        
        if 'coco' in self.root:
            self.db = self._get_db()

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_ann_file_keypoint(self):
        """ self.root / annotations / person_keypoints_train2017.json """
        if 'coco' in self.root:
            prefix = 'person_keypoints' \
                if 'test' not in self.image_set else 'image_info'
            return os.path.join(
                self.root,
                'annotations',
                prefix + '_' + self.image_set + '.json'
            )
        elif 'crowdpose' in self.root:
            prefix = 'crowdpose'
            return os.path.join(
                self.root,
                'json',
                prefix + '_' + self.image_set + '.json'
            )
        elif 'aic' in self.root:
            prefix = 'aic'
            return os.path.join(
                self.root,
                'annotations',
                prefix + '_' + self.image_set + '.json'
            )
        else:
            raise ValueError('Please write the path for this new dataset.')

    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_db(self):
        gt_db = self._load_coco_keypoint_annotations()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if 'crowdpose' in self.root:
                obj['area'] = 1
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                continue

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float32)
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            center, scale = self._box2cs(obj['clean_bbox'][:4])
            rec.append({
                'image': self.image_path_from_index(index, im_ann),
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
            })

        return rec

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def image_path_from_index(self, index, im_ann):
        """ example: images / train2017 / 000000119993.jpg """
        if 'coco' in self.root:
            file_name = '%012d.jpg' % index
            if '2014' in self.image_set:
                file_name = 'COCO_%s_' % self.image_set + file_name

            prefix = 'test2017' if 'test' in self.image_set else self.image_set

            data_name = prefix

            image_path = os.path.join(
                self.root, 'images', data_name, file_name)

            return image_path
        elif 'crowdpose' in self.root:
            file_name = f'{index}.jpg'

            image_path = os.path.join(
                self.root, 'images', file_name)

            return image_path
        elif 'aic' in self.root:
            file_name = im_ann["file_name"]

            image_path = os.path.join(
                self.root, 'ai_challenger_keypoint_train_20170902', 'keypoint_train_images_20170902', file_name)

            return image_path


def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


class CrowdPoseDataset(COCODataset):
    def __init__(self, root, image_set, is_train, max_prompt_num=5, min_prompt_num=1, 
            radius=10, size=256, transparency=0.0, sample_weight=1.0, transform=None):

        super().__init__(root, image_set, is_train, max_prompt_num, min_prompt_num, 
            radius, size, transparency, sample_weight, transform)

        self.keypoints_type = {
                0: 'left_shoulder',
                1: 'right_shoulder',
                2: 'left_elbow',
                3: 'right_elbow',
                4: 'left_wrist',
                5: 'right_wrist',
                6: 'left_hip',
                7: 'right_hip',
                8: 'left_knee',
                9: 'right_knee',
                10: 'left_ankle',
                11: 'right_ankle',
                12: 'top_head',
                13: 'neck'
            }
        
        self.num_joints = 14
        self.prob_half_body = -1
        self.flip_pairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]
        self.parent_ids = None
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 12, 13)
        self.lower_body_ids = (6, 7, 8, 9, 10, 11)

        self.db = self._get_db()

        logger.info('=> load {} samples'.format(len(self.db)))


class AICDataset(COCODataset):
    def __init__(self, root, image_set, is_train, max_prompt_num=5, min_prompt_num=1, 
            radius=10, size=256, transparency=0.0, sample_weight=1.0, transform=None):
        super().__init__(root, image_set, is_train, max_prompt_num, min_prompt_num, 
            radius, size, transparency, sample_weight, transform)

        self.keypoints_type = {
                0: "right_shoulder",
                1: "right_elbow",
                2: "right_wrist",
                3: "left_shoulder",
                4: "left_elbow",
                5: "left_wrist",
                6: "right_hip",
                7: "right_knee",
                8: "right_ankle",
                9: "left_hip",
                10: "left_knee",
                11: "left_ankle",
                12: "head_top",
                13: "neck"
            }
        
        self.num_joints = 14
        self.prob_half_body = -1
        self.flip_pairs = [[0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11]]
        self.parent_ids = None
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 12, 13)
        self.lower_body_ids = (6, 7, 8, 9, 10, 11)

        self.db = self._get_db()

        logger.info('=> load {} samples'.format(len(self.db)))


class MPIIDataset(PoseDataset):
    def __init__(self, root, image_set, is_train, max_prompt_num=5, min_prompt_num=1, 
            radius=10, size=256, transparency=0.0, sample_weight=1.0, transform=None):
        super().__init__(root, image_set, is_train, max_prompt_num, min_prompt_num, 
            radius, size, transparency, sample_weight, transform)

        self.keypoints_type = {
                0: 'right_ankle',
                1: 'right_knee',
                2: 'right_hip',
                3: 'left_hip',
                4: 'left_knee',
                5: 'left_ankle',
                6: 'pelvis',
                7: 'thorax',
                8: 'upper_neck',
                9: 'head_top',
                10: 'right_wrist',
                11: 'right_elbow',
                12: 'right_shoulder',
                13: 'left_shoulder',
                14: 'left_elbow',
                15: 'left_wrist'
            }
        
        self.data_format = 'jpg'
        self.num_joints = 16 
        self.prob_half_body = -1
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.parent_ids = None
        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)

        self.db = self._get_db()

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        # create train/val split
        file_name = os.path.join(
            self.root, 'annot', self.image_set+'.json'
        )
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=np.float32)
            s = np.array([a['scale'], a['scale']], dtype=np.float32)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float32)
            if self.image_set != 'test':
                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2] - 1
                joints_vis = np.array(a['joints_vis'])
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

            image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
            gt_db.append(
                {
                    'image': os.path.join(self.root, image_dir, image_name),
                    'center': c,
                    'scale': s,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                }
            )

        return gt_db
