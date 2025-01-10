import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


def main():
    img_size = 960

    # 创建目标文件夹
    os.makedirs(f'inputs/dsb2025_{img_size}/images', exist_ok=True)
    os.makedirs(f'inputs/dsb2025_{img_size}/masks/0', exist_ok=True)

    # 获取所有样本路径
    paths = glob('inputs/stage1_train/*')

    for i in tqdm(range(len(paths))):
        path = paths[i]

        # 读取图像文件
        image_path = glob(os.path.join(path, 'images', '*.png'))[0]
        img = cv2.imread(image_path)

        # 初始化空掩码
        mask = np.zeros((img.shape[0], img.shape[1]))

        # 读取所有掩码文件并合并
        for mask_path in glob(os.path.join(path, 'masks', '*')):
            mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127
            mask[mask_] = 1

        # 处理单通道和四通道图像
        if len(img.shape) == 2:
            img = np.tile(img[..., None], (1, 1, 3))
        if img.shape[2] == 4:
            img = img[..., :3]

        # 调整图像和掩码尺寸
        img = cv2.resize(img, (img_size, img_size))
        mask = cv2.resize(mask, (img_size, img_size))

        # 保存处理后的图像和掩码
        cv2.imwrite(os.path.join(f'inputs/dsb2025_{img_size}/images', f'sample{i+1}.png'), img)
        cv2.imwrite(os.path.join(f'inputs/dsb2025_{img_size}/masks/0', f'sample{i+1}.png'), (mask * 255).astype('uint8'))


if __name__ == '__main__':
    main()
