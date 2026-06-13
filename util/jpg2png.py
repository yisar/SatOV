import os
from PIL import Image

# 在这里填写你的图片文件夹路径
folder_path = r"./benchmark/UDD/origin"

for filename in os.listdir(folder_path):
    # 只处理 jpg/jpeg
    if filename.lower().endswith(('.jpg', '.jpeg')):
        img_path = os.path.join(folder_path, filename)
        try:
            with Image.open(img_path) as img:
                # 生成新文件名
                name_only, _ = os.path.splitext(filename)
                new_path = os.path.join(folder_path, f"{name_only}.png")
                img.save(new_path, 'PNG')
                print(f"已转换：{filename} -> {name_only}.png")

            # 删除原 jpg 文件
            os.remove(img_path)
            print(f"已删除原文件：{filename}")

        except Exception as e:
            print(f"处理失败 {filename}：{e}")

print("\n✅ 全部处理完成！")