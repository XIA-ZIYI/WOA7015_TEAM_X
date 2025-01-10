import os
import shutil
from PIL import Image

# 定义原始数据集路径和目标路径
src_images_path = "DRIVE/images"
src_masks_path = "DRIVE/mask"
dst_path = "inputs/stage1_train"

# 获取所有图像和掩码文件
image_files = sorted(os.listdir(src_images_path))
mask_files = sorted(os.listdir(src_masks_path))

# 确保图像和掩码文件数量一致
assert len(image_files) == len(mask_files), "图像和掩码文件数量不匹配！"

# 创建目标文件夹结构
os.makedirs(dst_path, exist_ok=True)

# 遍历图像和掩码文件，调整格式并移动到目标文件夹
for idx, (image_file, mask_file) in enumerate(zip(image_files, mask_files)):
    sample_folder = os.path.join(dst_path, f"sample{idx+1}")
    images_folder = os.path.join(sample_folder, "images")
    masks_folder = os.path.join(sample_folder, "masks")

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)

    # 复制并重命名图像文件
    img_src = os.path.join(src_images_path, image_file)
    img_dst = os.path.join(images_folder, f"sample{idx+1}.png")
    with Image.open(img_src) as img:
        img.convert("RGB").save(img_dst)

    # 复制并重命名掩码文件
    mask_src = os.path.join(src_masks_path, mask_file)
    mask_dst = os.path.join(masks_folder, "mask1.png")
    with Image.open(mask_src) as mask:
        mask.convert("L").save(mask_dst)

print("✅ 数据格式调整完成！")
