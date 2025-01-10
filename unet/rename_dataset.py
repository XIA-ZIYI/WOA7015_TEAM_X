import os
import shutil
from PIL import Image

# 定义路径
src_path = "/NEW_DATASET/gen"
dst_path = "/NEW_DATASET/gen_rename"

# 创建新的文件夹结构
os.makedirs(os.path.join(dst_path, "test", "images"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "test", "manual"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "test", "mask"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "training", "images"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "training", "manual"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "training", "mask"), exist_ok=True)

# 获取所有文件
image_files = sorted(os.listdir(os.path.join(src_path, "images")))
manual_files = sorted(os.listdir(os.path.join(src_path, "manual")))
mask_files = sorted(os.listdir(os.path.join(src_path, "mask")))

# 定义训练集和测试集的分界线
train_count = int(0.7 * len(image_files))

# 处理 images 文件
for idx, file_name in enumerate(image_files):
    img = Image.open(os.path.join(src_path, "images", file_name))
    new_name = f"{idx+1:02d}_{'training' if idx < train_count else 'test'}.tif"
    subfolder = "training" if idx < train_count else "test"
    img.save(os.path.join(dst_path, subfolder, "images", new_name))

# 处理 manual 文件
for idx, file_name in enumerate(manual_files):
    manual = Image.open(os.path.join(src_path, "manual", file_name))
    new_name = f"{idx+1:02d}_manual1.gif"
    subfolder = "training" if idx < train_count else "test"
    manual.save(os.path.join(dst_path, subfolder, "manual", new_name))

# 处理 mask 文件
for idx, file_name in enumerate(mask_files):
    mask = Image.open(os.path.join(src_path, "mask", file_name))
    new_name = f"{idx+1:02d}_{'training' if idx < train_count else 'test'}_mask.gif"
    subfolder = "training" if idx < train_count else "test"
    mask.save(os.path.join(dst_path, subfolder, "mask", new_name))

print("✅ 文件重命名和分类完成！")
