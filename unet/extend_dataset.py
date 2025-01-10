import os
from PIL import Image, ImageOps

# 定义源路径和目标路径
src_path = "/NEW_DATASET/DRIVE"
dst_path = "/NEW_DATASET/DRIVE_extend"

# 创建新的文件夹结构
for subdir in ["images", "manual", "mask"]:
    os.makedirs(os.path.join(dst_path, subdir), exist_ok=True)

# 数据扩充函数
def augment_image(img, rotation_interval):
    augmented_images = []

    # 1. 旋转
    for angle in range(rotation_interval, 360, rotation_interval):
        rotated_img = img.rotate(angle, resample=Image.BICUBIC)
        augmented_images.append(rotated_img)

    # 2. 翻转
    for rotated_img in augmented_images.copy():
        augmented_images.append(ImageOps.mirror(rotated_img))  # 水平翻转
        augmented_images.append(ImageOps.flip(rotated_img))  # 垂直翻转

    return augmented_images

# 定义旋转角度间隔
rotation_intervals = {"test": 22, "training": 5}

# 处理每个文件夹
for subfolder in ["test", "training"]:
    for subdir in ["images", "manual", "mask"]:
        src_subdir = os.path.join(src_path, subfolder, subdir)
        dst_subdir = os.path.join(dst_path, subdir)

        original_files = sorted(os.listdir(src_subdir))
        count = len(os.listdir(dst_subdir)) + 1

        for idx, file_name in enumerate(original_files):
            img = Image.open(os.path.join(src_subdir, file_name))
            base_name, ext = os.path.splitext(file_name)

            # 保存原图
            img.save(os.path.join(dst_subdir, f"{count:04d}{ext}"))

            # 生成扩充后的图像
            augmented_images = augment_image(img, rotation_intervals[subfolder])

            # 保存扩充后的图像
            for aug_idx, aug_img in enumerate(augmented_images, start=1):
                aug_img.save(os.path.join(dst_subdir, f"{count + aug_idx:04d}{ext}"))

            count += len(augmented_images) + 1

print("✅ 数据扩容完成！")
