# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
import shutil
import tempfile
import zipfile

import mmcv
import numpy as np
from PIL import Image

okutama_drone_palette = \
    {
        0: (0, 0, 0),
        1: (1, 1, 1),
        2: (2, 2, 2),
        3: (3, 3, 3),
        4: (4, 4, 4),
        5: (5, 5, 5),
        6: (6, 6, 6),
        7: (7, 7, 7),
        8: (8, 8, 8),
        9: (9, 9, 9)
    }

okutama_drone_invert_palette = {v: k for k, v in okutama_drone_palette.items()}


def imgAnn_convert_from_color(arr_3d, palette=okutama_drone_invert_palette):
    """RGB-color encoding to grayscale labels."""
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    # print(f"arr_3d.shape: {arr_3d.shape}")
    # print(f"arr_2d.shape: {arr_2d.shape}")
    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def fix_labels(src_path, out_dir, mode):
    label = mmcv.imread(src_path, channel_order='rgb')
    # print(f"\nlabel.shape: {label.shape}")
    label = imgAnn_convert_from_color(label)
    # print(f"label.shape: {label.shape}")
    converted_label_img = Image.fromarray(label.astype(np.uint8), mode='P')
    # print(f"converted_label_img.shape: {converted_label_img.size}")
    converted_label_img.save(osp.join(out_dir, 'ann_dir', mode, osp.basename(src_path)))
                                      


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Okutama-Swiss Drone dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='Okutama-Swiss zip file path')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path

    if args.out_dir is None:
        out_dir = osp.join('data', 'Okutama-Swiss-dataset')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'test'))

    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'test'))

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        zip_file = zipfile.ZipFile(dataset_path)
        zip_file.extractall(tmp_dir)
        
        for dataset_mode in ['train', 'val', 'test']:
            print('\nProcessing {} images...'.format(dataset_mode))
            img_list = glob.glob(
                os.path.join(tmp_dir, 'images', dataset_mode, '*.png')) + glob.glob(
                os.path.join(tmp_dir, 'images', dataset_mode, '*.JPG'))
            # print('Find the data', img_list)
            src_prog_bar = mmcv.ProgressBar(len(img_list))
            for img in img_list:
                shutil.copy(img, os.path.join(out_dir, 'img_dir', dataset_mode))
                src_prog_bar.update()

            print('\nProcessing {} annotations...'.format(dataset_mode))
            # do not skip test as we have annotations for test set! not done!
            # if dataset_mode != 'test':
            label_list = glob.glob(
                os.path.join(tmp_dir, 'ground_truth', dataset_mode, '*.png'))
            lab_prog_bar = mmcv.ProgressBar(len(label_list))
            for label in label_list:
                label = fix_labels(label, out_dir, dataset_mode)
                lab_prog_bar.update()
            # for label in label_list:
            #     shutil.copy(label, os.path.join(out_dir, 'ann_dir', dataset_mode))


        print('Removing the temporary files...')

    print('Done!')


if __name__ == '__main__':
    main()
