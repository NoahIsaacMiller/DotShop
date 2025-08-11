from modulator import DisplayDataGenerator
from data_structures import PixelMatrix, ScreenConfig, ColorMode, ScanDirection
from PIL import Image
import pygame

# 
img = Image.open("/Users/noahmiller/Code/DotShop/dotshop/core/20220302224226_243a0.gif").copy()

width = 240
height = 320

frame = img.resize((width, height))

# 使用二值化后的图片创建像素矩阵
matrix = PixelMatrix.from_image(frame, ColorMode.RGB565)

screenConfig = ScreenConfig(width, height, [ColorMode.RGB565], ScanDirection.HORIZONTAL)
# 初始化调制器
modulator = DisplayDataGenerator(screenConfig)

# 取模处理
data = list(byte for byte in modulator.generate(matrix))

c_style = """\
#include<stdint.h>
int ImageWidth = {};
int ImageHeight = {};
int bytePrePixel = {};
uint8_t images[] = {};
"""

bytesArray = "{" + ", ".join([f"0x{byte:02x}" for byte in data]) + "}"
with open("image.h", "w", encoding="utf8") as f:
    text = c_style.format(width, height, 2, bytesArray)
    print(text, file=f)