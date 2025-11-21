import sys
import numpy as np

def read_ppm(filename):
    """读取PPM文件并返回图像数据"""
    with open(filename, 'rb') as f:
        # 读取魔数
        magic = f.readline().decode().strip()
        
        # 跳过注释行
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        
        # 读取宽度和高度
        width, height = map(int, line.decode().split())
        
        # 读取最大颜色值
        max_val = int(f.readline().decode().strip())
        
        # 读取像素数据
        if magic == 'P3':  # ASCII格式
            data = []
            for line in f:
                data.extend(map(int, line.decode().split()))
            pixels = np.array(data, dtype=np.uint8).reshape((height, width, 3))
        elif magic == 'P6':  # 二进制格式
            pixels = np.frombuffer(f.read(), dtype=np.uint8).reshape((height, width, 3))
        else:
            raise ValueError(f"不支持的PPM格式: {magic}")
        
        return pixels, width, height, max_val

def compare_images(file1, file2):
    """对比两个PPM图像的像素差异"""
    print(f"读取文件: {file1}")
    img1, w1, h1, max1 = read_ppm(file1)
    
    print(f"读取文件: {file2}")
    img2, w2, h2, max2 = read_ppm(file2)
    
    # 检查尺寸是否一致
    if (w1, h1) != (w2, h2):
        print(f"错误: 图像尺寸不匹配! {w1}x{h1} vs {w2}x{h2}")
        return
    
    print(f"\n图像尺寸: {w1} x {h1}")
    print(f"最大颜色值: {max1} vs {max2}")
    
    # 计算差异
    diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
    
    # 统计信息
    total_pixels = w1 * h1
    different_pixels = np.sum(np.any(diff > 1, axis=2))
    identical_pixels = total_pixels - different_pixels
    
    print(f"\n=== 像素对比结果 ===")
    print(f"总像素数: {total_pixels}")
    print(f"相同像素数: {identical_pixels} ({identical_pixels/total_pixels*100:.2f}%)")
    print(f"不同像素数: {different_pixels} ({different_pixels/total_pixels*100:.2f}%)")
    
    if different_pixels > 0:
        print(f"\n=== 差异统计 ===")
        print(f"最大差异 (R): {np.max(diff[:,:,0])}")
        print(f"最大差异 (G): {np.max(diff[:,:,1])}")
        print(f"最大差异 (B): {np.max(diff[:,:,2])}")
        print(f"平均差异 (R): {np.mean(diff[:,:,0]):.2f}")
        print(f"平均差异 (G): {np.mean(diff[:,:,1]):.2f}")
        print(f"平均差异 (B): {np.mean(diff[:,:,2]):.2f}")
        
        # 显示前10个不同的像素位置
        print(f"\n=== 前10个不同像素的位置 ===")
        diff_positions = np.argwhere(np.any(diff > 1, axis=2))
        for i, (y, x) in enumerate(diff_positions[:10]):
            print(f"位置 ({x}, {y}): "
                  f"图1 RGB{tuple(img1[y,x])} vs 图2 RGB{tuple(img2[y,x])} "
                  f"差异 {tuple(diff[y,x])}")
    else:
        print("\n两个图像完全相同!")

if __name__ == "__main__":
    file1 = "render/cuda_fuck_0000.ppm"
    file2 = "render/ref_fuck_0000.ppm"
    
    try:
        compare_images(file1, file2)
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 - {e}")
    except Exception as e:
        print(f"错误: {e}")
