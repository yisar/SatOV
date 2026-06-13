import os
import matplotlib.pyplot as plt
from PIL import Image


def plot_2row3col_with_col_title(
    image_folder, prefix_list, save_path="result.png", dpi=300
):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))

    for col_idx, prefix in enumerate(prefix_list):
        ori_path = os.path.join(image_folder, f"{prefix}.png")
        mask_path = os.path.join(image_folder, f"{prefix}-mask.png")

        # 文件不存在则提示
        if not os.path.exists(ori_path):
            raise FileNotFoundError(f"原图不存在：{ori_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"掩码图不存在：{mask_path}")

        img_ori = Image.open(ori_path).convert("RGB")
        axes[0, col_idx].imshow(img_ori)
        axes[0, col_idx].set_title(prefix, fontsize=11)
        axes[0, col_idx].axis("off")

        img_mask = Image.open(mask_path).convert("RGB")
        axes[1, col_idx].imshow(img_mask)
        axes[1, col_idx].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # 统一用绝对路径更稳妥，避免相对路径出错
    folder = r"./bench/demo"
    prefixs = ["Dense Scene", "Farmland & Trees", "Rotation Invariance"]
    try:
        plot_2row3col_with_col_title(folder, prefixs)
    except FileNotFoundError as e:
        print(e)
        # 打印当前文件夹下所有文件，方便核对文件名
        print("当前文件夹内所有文件：")
        for f in os.listdir(folder):
            print(f)
