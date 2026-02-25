import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import median_filter
import datetime


def repair_depth_noise_focused(
    depth_img,
    max_valid_depth=10000,
    median_kernel=5,
    detect_white_spots=True,
    spot_size_range=(10, 1000),
):
    """
    专门针对黑色背景下白色圆斑噪点的修复算法（16位原生处理）
    """
    # log_print(f"[DEBUG] 开始检测白色圆斑噪点，图像形状: {depth_img.shape}")
    # log_print(f"[DEBUG] 深度值范围: {depth_img.min()} - {depth_img.max()}")
    # log_print(f"[DEBUG] 图像数据类型: {depth_img.dtype}")

    # 1. 检测超远距离噪点
    distance_noise_mask = depth_img >= max_valid_depth

    # 2. 专门检测黑色背景下的白色圆斑（直接在16位上处理）
    white_spot_mask = np.zeros_like(depth_img, dtype=bool)

    if detect_white_spots:
        # 方法1: 直接在16位深度图上检测异常高值的小圆形区域
        valid_depths = depth_img[depth_img >= 0]
        if len(valid_depths) > 100:  # 确保有足够的有效像素
            mean_depth = np.mean(valid_depths)
            std_depth = np.std(valid_depths)

            # 设置阈值：比平均值高2个标准差的区域
            high_depth_threshold = mean_depth + 0.2 * std_depth  # 从1增强到0.05
            # log_print(f"[DEBUG] 深度统计: 均值={mean_depth:.1f}, 标准差={std_depth:.1f}, 高值阈值={high_depth_threshold:.1f}")

            # 检测高深度值区域
            high_depth_mask = depth_img > high_depth_threshold

            # 使用连通域分析找出小的高深度值区域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                high_depth_mask.astype(np.uint8), connectivity=4
            )

            for i in range(1, num_labels):  # 跳过背景标签0
                area = stats[i, cv2.CC_STAT_AREA]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]

                # 检查是否符合白色圆斑特征
                if (
                    spot_size_range[0] <= area <= spot_size_range[1]
                ):  # 接近圆形（放宽一点）

                    region_mask = labels == i
                    white_spot_mask |= region_mask

        # 方法2: 检测局部极大值（在16位上直接操作）
        # 使用形态学操作检测局部极大值
        structuring_element = np.ones((9, 9), dtype=np.uint8)  # 9x9的结构元素
        dilated = cv2.dilate(depth_img, structuring_element, iterations=1)

        # 找出局部极大值（原图等于膨胀后的图像的点）
        local_maxima = (depth_img == dilated) & (depth_img >= 0)

        # 过滤掉不够"突出"的极大值
        if len(valid_depths) >= 0:  # 确保有有效深度数据
            # 计算每个像素与其邻域的差异
            kernel_avg = np.ones((15, 15), dtype=np.float32) / (15 * 15)
            neighborhood_avg = cv2.filter2D(
                depth_img.astype(np.float32), -1, kernel_avg
            )

            # 找出比邻域平均值高很多的点
            significant_peaks = local_maxima & (
                depth_img > neighborhood_avg + std_depth
            )

            # 对这些峰值点进行连通域分析
            if np.sum(significant_peaks) > 0:
                # 修正：正确解包cv2.connectedComponents的返回值
                peak_num, peak_labels = cv2.connectedComponents(
                    significant_peaks.astype(np.uint8)
                )

                for i in range(1, peak_num):  # 修正：从1到peak_num-1
                    peak_region = peak_labels == i
                    area = np.sum(peak_region)

                    if spot_size_range[0] <= area <= spot_size_range[1]:
                        white_spot_mask |= peak_region

    # 3. 合并所有类型的噪点
    noise_mask = distance_noise_mask | white_spot_mask

    distance_noise_count = np.sum(distance_noise_mask)
    white_spot_count = np.sum(white_spot_mask)
    total_noise_count = np.sum(noise_mask)

    # log_print(f"[DEBUG] 距离噪点: {distance_noise_count} 像素")
    # log_print(f"[DEBUG] 白色圆斑噪点: {white_spot_count} 像素")
    # log_print(f"[DEBUG] 总噪点: {total_noise_count} 像素")

    if total_noise_count == 0:
        return depth_img.copy()

    # 4. 16位原生修复：使用scipy的median_filter（支持16位）
    repaired_img = depth_img.copy()

    # 确保median_kernel是奇数
    if median_kernel % 2 == 0:
        median_kernel += 1

    # 对白色圆斑使用更大的中值滤波核
    if white_spot_count > 0:
        # 找出每个独立的白色圆斑区域
        spot_num, spot_labels = cv2.connectedComponents(
            white_spot_mask.astype(np.uint8)
        )

        for spot_id in range(1, spot_num):
            spot_region = spot_labels == spot_id

            # 创建比圆斑稍大的邻域区域
            dilate_kernel = np.ones((5, 5), dtype=np.uint8)  # 5x5扩展
            expanded_region = cv2.dilate(
                spot_region.astype(np.uint8), dilate_kernel, iterations=2
            )

            # 外围环形区域 = 扩展区域 - 原圆斑
            outer_ring = (expanded_region.astype(bool)) & (~spot_region)

            # 计算外围区域的均值
            outer_values = depth_img[outer_ring & (depth_img >= 0)]

            if len(outer_values) > 3:  # 至少3个有效值
                replacement_value = int(np.mean(outer_values))
            else:
                # 备选：使用全局有效像素均值
                replacement_value = int(np.mean(valid_depths))

            # 用均值替代整个圆斑
            repaired_img[spot_region] = replacement_value

    # 对距离噪点使用常规中值滤波
    distance_only_mask = distance_noise_mask & ~white_spot_mask
    if np.sum(distance_only_mask) > 0:
        median_filtered = median_filter(depth_img, size=median_kernel)
        repaired_img[distance_only_mask] = median_filtered[distance_only_mask]
        # log_print(f"[DEBUG] 距离噪点使用 {median_kernel}x{median_kernel} 中值滤波修复")

    # log_print(f"[DEBUG] 16位原生修复完成，无精度损失")

    return repaired_img


def visualize_repair_comparison_enhanced(
    original_img, repaired_img, noise_mask, save_path=None
):
    """
    增强版可视化，突出显示白色圆斑检测
    """
    # 创建彩色深度图
    original_colored = create_depth_colormap(original_img)
    repaired_colored = create_depth_colormap(repaired_img)

    # 创建噪点可视化，用不同颜色标记不同类型噪点
    noise_overlay = original_colored.copy()

    # 距离噪点标记为红色
    distance_noise = original_img >= 10000
    noise_overlay[distance_noise] = [255, 0, 0]  # 红色

    # 其他噪点（主要是白色圆斑）标记为黄色
    other_noise = noise_mask & ~distance_noise
    noise_overlay[other_noise] = [255, 255, 0]  # 黄色

    # 绘制对比图 - 改为7个子图
    plt.figure(figsize=(28, 6))

    # 1. 原始灰度图
    plt.subplot(1, 7, 1)
    plt.imshow(original_img, cmap="gray")
    plt.title("Original Depth\n(Grayscale)")
    plt.colorbar(label="Depth Value")
    plt.axis("off")

    # 2. 修复后灰度图 - 新添加
    plt.subplot(1, 7, 2)
    plt.imshow(repaired_img, cmap="gray")
    plt.title("Repaired Depth\n(Grayscale)")
    plt.colorbar(label="Depth Value")
    plt.axis("off")

    # 3. 原始彩色图
    plt.subplot(1, 7, 3)
    plt.imshow(original_colored)
    plt.title("Original Depth\n(Red-Blue)")
    plt.axis("off")

    # 4. 噪点检测（分类显示）
    plt.subplot(1, 7, 4)
    plt.imshow(noise_overlay)
    distance_count = np.sum(distance_noise)
    spot_count = np.sum(other_noise)
    plt.title(
        f"Noise Detection\nRed: Distance({distance_count})\nYellow: Spots({spot_count})"
    )
    plt.axis("off")

    # 5. 修复后彩色图
    plt.subplot(1, 7, 5)
    plt.imshow(repaired_colored)
    plt.title("Repaired Depth\n(Red-Blue)")
    plt.axis("off")

    # 6. 差异图
    plt.subplot(1, 7, 6)
    diff = np.abs(repaired_img.astype(np.int32) - original_img.astype(np.int32))
    plt.imshow(diff, cmap="hot")
    plt.title("Difference Map\n(Bright=Large Change)")
    plt.colorbar(label="Difference")
    plt.axis("off")

    # 7. 噪点区域的局部放大图
    plt.subplot(1, 7, 7)
    if np.sum(noise_mask) > 0:
        noise_coords = np.where(noise_mask)
        y_min, y_max = max(0, noise_coords[0].min() - 20), min(
            original_img.shape[0], noise_coords[0].max() + 20
        )
        x_min, x_max = max(0, noise_coords[1].min() - 20), min(
            original_img.shape[1], noise_coords[1].max() + 20
        )

        crop_original = original_img[y_min:y_max, x_min:x_max]
        plt.imshow(crop_original, cmap="gray")
        plt.title(f"Noise Region Zoom\n({x_max-x_min}x{y_max-y_min})")
        plt.axis("off")
    else:
        plt.text(
            0.5,
            0.5,
            "No Noise\nDetected",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
        plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log_print(f"[INFO] Enhanced comparison saved to: {save_path}")

    plt.show()


def create_depth_colormap(depth_img):
    """
    创建深度图的红蓝色彩映射（排除0值）
    """
    # 使用非零值的最小最大值归一化
    valid_pixels = depth_img[depth_img > 0]
    if len(valid_pixels) == 0:
        return np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)

    min_depth = np.min(valid_pixels)
    max_depth = np.max(valid_pixels)

    if max_depth == min_depth:
        depth_normalized = np.zeros_like(depth_img, dtype=np.float32)
    else:
        depth_normalized = (depth_img.astype(np.float32) - min_depth) / (
            max_depth - min_depth
        )
        depth_normalized = np.clip(depth_normalized, 0, 1)

    # 创建RGB图像
    colored_depth = np.zeros(
        (depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8
    )

    # 只对有效像素（非零）进行着色
    valid_mask = depth_img > 0
    colored_depth[valid_mask, 0] = (255 * (1 - depth_normalized[valid_mask])).astype(
        np.uint8
    )  # Red
    colored_depth[valid_mask, 2] = (255 * depth_normalized[valid_mask]).astype(
        np.uint8
    )  # Blue

    return colored_depth


def test_white_spot_repair(input_png_path):
    """
    专门测试白色圆斑修复
    """
    log_print("=" * 70)
    log_print("开始白色圆斑噪点检测与修复测试（16位原生处理）")
    log_print("=" * 70)

    # 读取图像
    depth_img = cv2.imread(input_png_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        log_print(f"[ERROR] 无法读取文件: {input_png_path}")
        return

    if depth_img.ndim > 2:
        depth_img = depth_img[:, :, 0]
    if depth_img.dtype != np.uint16:
        depth_img = depth_img.astype(np.uint16)

    # 修复噪点
    starttime = datetime.datetime.now()
    repaired_img, noise_mask = repair_depth_noise_focused(
        depth_img,
        max_valid_depth=10000,
        median_kernel=5,
        detect_white_spots=True,
        spot_size_range=(10, 1000),  # 调整圆斑大小范围
    )
    endtime = datetime.datetime.now()
    log_print(f"修复耗时: {endtime - starttime}")
    # 保存修复后的图像
    output_path = input_png_path.replace(".png", "_spot_repaired.png")
    cv2.imwrite(output_path, repaired_img)

    # 增强可视化
    comparison_path = input_png_path.replace(".png", "_spot_comparison.png")
    visualize_repair_comparison_enhanced(
        depth_img, repaired_img, noise_mask, comparison_path
    )

    repaired_pixels = np.sum(noise_mask)
    log_print(f"✅ 白色圆斑修复成功！修复了 {repaired_pixels} 个噪点像素")
    log_print(f"💾 修复图像已保存到: {output_path}")
    log_print(f"📊 增强对比图已显示并保存")


# 使用方法
if __name__ == "__main__":
    test_white_spot_repair("test_mkv/output_0022.png")
