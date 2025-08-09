from modulator import Modulator
from data_structures import PixelMatrix, ScreenConfig, ScreenType, ColorMode, ScanDirection
from PIL import Image, ImageOps
import pygame
import time

def binarize_image(image, threshold=128):
    """
    将图片转换为灰度图后进行二值化处理
    
    Args:
        image: PIL Image对象
        threshold: 二值化阈值，默认128，大于此值的像素变为白色(255)，否则为黑色(0)
        
    Returns:
        二值化处理后的PIL Image对象
    """
    # 转换为灰度图
    gray_image = ImageOps.grayscale(image)
    
    # 二值化处理
    binary_image = gray_image.point(lambda x: 1 if x > threshold else 0, '1')

    
    return binary_image

# 图片路径
path = "/Users/noahmiller/Code/DotShop/dotshop/core/20220302224226_243a0.gif"
img = Image.open(path)

# 先进行二值化处理
binary_img = binarize_image(img, threshold=128)  # 可根据需要调整阈值
# binary_img.show()

# 使用二值化后的图片创建像素矩阵
matrix = PixelMatrix.from_image(binary_img.resize((128, 128)), ColorMode.MONO)

# 初始化调制器
modulator = Modulator(ScreenConfig(128, 128, ScanDirection.VERTICAL_BY_PAGE))

# 取模处理
mode = list(byte for byte in modulator.modulate(matrix))
# exit()
# 初始化Pygame显示
pygame.init()

width = 128
height = 128

screen = pygame.display.set_mode(
    (width, height), 
    flags=pygame.NOFRAME | pygame.DOUBLEBUF
)

def modeOnScreen(screen: pygame.Surface, mode_data):
    """将取模数据显示在屏幕上"""
    for page in range(8):
        for col in range(width):
            # 计算当前字节索引
            index = page * width + col
            if index >= len(mode_data):
                continue
                
            byte = mode_data[index]
            # 处理每个字节的8位
            for i in range(8):
                # 计算当前像素位置
                y_pos = page * 8 + i
                if y_pos >= height:
                    continue
                    
                # 检查当前位是否设置
                if byte & (0x80 >> i):
                    rect = pygame.Rect(col, y_pos, 1, 1)
                    pygame.draw.rect(screen, "white", rect)

# def modeOnScreen(screen: pygame.Surface, mode_data):
#     """将取模数据显示在屏幕上"""
#     for x in range(width):
#         for y in range(height):
#             if mode_data[y * width + x]:
#                 pygame.draw.rect(screen, "white", pygame.Rect(x, y, 1, 1))

# 主循环
running = True
while running:
    # 事件处理
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:  # 按Q键退出
                running = False
    
    # 清屏
    screen.fill("black")
    
    # 显示取模数据
    modeOnScreen(screen, mode)
    
    # 更新显示
    pygame.display.update()
    
    # 短暂延迟
    time.sleep(0.1)

# 退出Pygame
pygame.quit()
