import argparse
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.core.composition import Compose
from albumentations import Resize, Normalize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter

"""
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    with open('models/data_NestedUNet_woDS/config.yml', 'r') as f:
        # with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    # 创建模型
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # 数据加载
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    # 加载模型权重
    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    # 定义验证集的变换
    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # 计算输出
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            iou = iou_score(output, target)
            avg_meter.update(iou, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % avg_meter.avg)

    # 获取一个批次的数据用于展示
    sample_inputs, sample_targets, _ = next(iter(val_loader))
    plot_examples(sample_inputs, sample_targets, model, num_examples=3, config=config)

    torch.cuda.empty_cache()


def plot_examples(datax, datay, model, num_examples=3, config=None):
    model.eval()
    fig, axes = plt.subplots(nrows=num_examples, ncols=3, figsize=(18, 6 * num_examples))

    if num_examples == 1:
        axes = [axes]  # 确保 axes 是一个列表，即使只有一个样本

    # 定义反归一化参数（与 albumentations.Normalize() 一致）
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        m = datax.shape[0]
        for row_num in range(num_examples):
            image_indx = row_num  # 使用前num_examples个样本
            input_image = datax[image_indx:image_indx + 1].cuda()
            output = model(input_image)
            if isinstance(output, list):
                output = output[-1]
            output = torch.sigmoid(output).squeeze(0).cpu().numpy()
            output = (output > 0.40).astype(int)

            # 原始图像 (RGB) 反归一化
            orig_img = datax[image_indx].cpu().numpy().transpose(1, 2, 0)
            orig_img = (orig_img * std + mean)  # 反归一化
            orig_img = np.clip(orig_img, 0, 1)  # 确保像素值在 [0, 1] 范围内

            # 转换为 RGB 格式（如果需要）
            # 如果原始图像是BGR格式（由OpenCV读取），需要转换为RGB
            # orig_img = orig_img[..., ::-1]

            # 目标图 (mask)
            target_img = datay[image_indx].cpu().numpy().transpose(1, 2, 0)
            target_img = target_img.squeeze(-1)  # 如果是单通道
            target_img = (target_img > 0.5).astype(int)

            # 预测图
            pred_img = output.squeeze()

            # 显示原始图像
            axes[row_num][0].imshow(orig_img)
            axes[row_num][0].set_title("Original RGB Image")
            axes[row_num][0].axis('off')

            # 显示预测的分割图
            axes[row_num][1].imshow(pred_img, cmap='gray')
            axes[row_num][1].set_title("Predicted Segmentation")
            axes[row_num][1].axis('off')

            # 显示目标分割图
            axes[row_num][2].imshow(target_img, cmap='gray')
            axes[row_num][2].set_title("Target Segmentation")
            axes[row_num][2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
