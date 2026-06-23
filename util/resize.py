from PIL import Image
import os

def resize_images_fixed_width(input_dir, output_dir, target_width=1000):
    # 创建输出文件夹（不存在则新建）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 支持的图片后缀
    img_suffix = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    # 遍历输入文件夹所有文件
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        # 跳过文件夹，只处理图片文件
        if os.path.isdir(file_path):
            continue
        if not filename.lower().endswith(img_suffix):
            continue

        try:
            # 打开图片
            img = Image.open(file_path)
            ori_w, ori_h = img.size

            # 计算等比例高度
            scale = target_width / ori_w
            target_height = int(ori_h * scale)

            # 高质量缩放
            new_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

            # 保存到输出文件夹
            out_path = os.path.join(output_dir, filename)
            # 保留原图质量，png自动无损
            if filename.lower().endswith(".png"):
                new_img.save(out_path, format="PNG")
            else:
                new_img.save(out_path, quality=95, optimize=True)

            print(f"处理完成：{filename} | 尺寸 {ori_w}×{ori_h} → {target_width}×{target_height}")

        except Exception as e:
            print(f"处理失败 {filename}，错误：{str(e)}")

if __name__ == "__main__":
    # ========== 修改这里的路径 ==========
    INPUT_FOLDER = r"./out/gt"   # 输入图片文件夹
    OUTPUT_FOLDER = r"./out/UUD6_gt" # 输出保存文件夹
    FIX_WIDTH = 1000            # 固定宽度

    resize_images_fixed_width(INPUT_FOLDER, OUTPUT_FOLDER, FIX_WIDTH)