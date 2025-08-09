from typing import Literal, Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
import copy
from PIL import Image


# ------------------------------
# 枚举定义: 替换魔法值
# ------------------------------
class ColorMode(Enum):
    """颜色模式枚举,定义支持的图像色彩格式"""
    MONO = auto()          # 二值模式(黑白)
    GRAY_8 = auto()        # 8位灰度模式
    RGB_888 = auto()       # 24位RGB模式(红、绿、蓝各8位)
    RGB_8888 = auto()      # 32位RGBA模式(含Alpha通道)
    RGB565 = auto()        # 16位RGB565模式(红5位、绿6位、蓝5位)

    def __str__(self) -> str:
        """返回小写字符串表示,与原始魔法值兼容"""
        return self.name.lower()


class ScanDirection(Enum):
    """扫描方向枚举,定义像素数据在屏幕上的读写顺序"""
    HORIZONTAL = auto()                     # 水平扫描(从左到右)
    VERTICAL = auto()                       # 垂直扫描(从上到下)
    HORIZONTAL_REVERSED = auto()            # 水平反向扫描(从右到左)
    VERTICAL_REVERSED = auto()              # 垂直反向扫描(从下到上)
    VERTICAL_BY_PAGE = auto()               # 按页垂直扫描
    VERTICAL_REVERSED_BY_PAGE = auto()      # 按页垂直反向扫描

    def __str__(self) -> str:
        """返回小写字符串表示,与原始魔法值兼容"""
        return self.name.lower()


class BitOrder(Enum):
    """位序枚举,定义屏幕硬件处理像素数据的位优先级"""
    MSB_FIRST = auto()  # 高位优先(Most Significant Bit First)
    LSB_FIRST = auto()  # 低位优先(Least Significant Bit First)

    def __str__(self) -> str:
        """返回小写字符串表示,与原始魔法值兼容"""
        return self.name.lower()


class ScreenType(Enum):
    """屏幕类型枚举,区分不同硬件特性"""
    LCD = auto()        # 液晶显示器
    OLED = auto()       # 有机发光二极管显示器
    E_PAPER = auto()    # 电子纸显示器
    LED_MATRIX = auto() # LED矩阵显示器

    def __str__(self) -> str:
        """返回小写字符串表示,与原始魔法值兼容"""
        return self.name.lower()


# ------------------------------
# 类型定义
# ------------------------------
ColorType = Union[bool, int, Tuple[int, ...]]  # 支持的颜色输入类型
CoordinateType = Tuple[int, int]  # 坐标类型定义(x, y)


# ------------------------------
# 模式配置注册表: 定义每种模式的数组结构
# ------------------------------
_VALID_MODES = {
    ColorMode.MONO: {
        "ndim": 2, 
        "dtype": np.bool_, 
        "description": "二值模式: 2维布尔数组,True表示白色,False表示黑色"
    },
    ColorMode.GRAY_8: {
        "ndim": 2, 
        "dtype": np.uint8, 
        "description": "8位灰度模式: 2维无符号整数数组,值范围0-255"
    },
    ColorMode.RGB_888: {
        "ndim": 3, 
        "dtype": np.uint8, 
        "channels": 3,
        "description": "24位RGB模式: 3维数组,形状为(height, width, 3),分别对应R、G、B通道"
    },
    ColorMode.RGB_8888: {
        "ndim": 3, 
        "dtype": np.uint8, 
        "channels": 4,
        "description": "32位RGBA模式: 3维数组,形状为(height, width, 4),包含R、G、B、Alpha通道"
    },
    ColorMode.RGB565: {
        "ndim": 2, 
        "dtype": np.uint16,
        "description": "16位RGB565模式: 2维无符号整数数组,每个值包含R(5位)、G(6位)、B(5位)信息"
    }
}


# ------------------------------
# 模式验证工具函数
# ------------------------------
def validate_mode(mode: ColorMode) -> bool:
    """验证模式是否为支持的颜色模式
    
    Args:
        mode: 待验证的颜色模式
        
    Returns:
        若模式支持则返回True,否则返回False
    """
    return mode in _VALID_MODES


def validate_mode_ex(mode: ColorMode) -> None:
    """验证模式合法性,不支持则抛出异常
    
    Args:
        mode: 待验证的颜色模式
        
    Raises:
        ValueError: 当模式不被支持时抛出
    """
    if not validate_mode(mode):
        supported_modes = [str(m) for m in _VALID_MODES.keys()]
        raise ValueError(
            f"不支持的模式: {mode},支持的模式为: {supported_modes}"
        )


# ------------------------------
# 像素矩阵类: 管理不同色彩模式的像素数据
# ------------------------------
class PixelMatrix:
    """像素矩阵容器,统一管理不同色彩模式的像素数据存储与访问
    
    该类封装了底层numpy数组,提供像素级操作接口,确保数据格式与指定的
    颜色模式一致,并处理模式间的转换逻辑。
    numpy数组需要严格按照Array[y][x]的方式排列
    """
    __width: int  # 矩阵宽度(像素)
    __height: int  # 矩阵高度(像素)
    __matrix: np.ndarray  # 底层numpy数组存储
    __mode: ColorMode  # 当前图像模式

    def __init__(self, matrix: np.ndarray, mode: ColorMode):
        """初始化像素矩阵并验证数据合法性
        
        Args:
            matrix: 底层像素数据的numpy数组
            mode: 图像模式(需为ColorMode枚举成员)
            
        Raises:
            ValueError: 模式不支持或矩阵参数不匹配
            TypeError: 矩阵数据类型与模式要求不符
        """
        # 验证模式合法性
        validate_mode_ex(mode)
        
        mode_config = _VALID_MODES[mode]
        
        # 验证矩阵维度
        if matrix.ndim != mode_config["ndim"]:
            raise ValueError(
                f"模式 {mode} 要求矩阵维度为 {mode_config['ndim']}, 但实际为 {matrix.ndim}"
            )
        
        # 验证数据类型
        if matrix.dtype != mode_config["dtype"]:
            raise TypeError(
                f"模式 {mode} 要求数据类型为 {mode_config['dtype']}, 但实际为 {matrix.dtype}"
            )
        
        # 验证多通道模式的通道数
        if mode_config["ndim"] == 3:
            expected_channels = mode_config.get("channels", 3)
            if matrix.shape[2] != expected_channels:
                raise ValueError(
                    f"模式 {mode} 要求通道数为 {expected_channels}, 但实际为 {matrix.shape[2]}"
                )
        
        # 初始化属性
        self.__mode = mode
        self.__matrix = matrix
        self.__height, self.__width = matrix.shape[:2]  # 前两维固定为高和宽

    @property
    def mode(self) -> ColorMode:
        """返回当前图像模式"""
        return self.__mode

    @property
    def width(self) -> int:
        """返回矩阵宽度(像素)"""
        return self.__width

    @property
    def height(self) -> int:
        """返回矩阵高度(像素)"""
        return self.__height

    @property
    def matrix(self) -> np.ndarray:
        """返回矩阵副本(避免外部直接修改内部数据)"""
        return self.__matrix.copy()

    def get_pixel(self, x: int, y: int) -> Tuple[int, ...]:
        """获取指定坐标的像素值,统一返回元组格式
        
        Args:
            x: 像素x坐标(水平方向)
            y: 像素y坐标(垂直方向)
            
        Returns:
            像素值元组(单通道模式返回长度为1的元组,多通道按通道顺序返回)
            
        Raises:
            IndexError: 当坐标超出矩阵范围时抛出
        """
        self.__is_coordinate_out_of_range_ex(x, y)
        
        pixel = self.__matrix[x, y]
        
        # 统一转换为元组格式输出
        if not isinstance(pixel, (tuple, np.ndarray)):
            return (pixel,)
        return tuple(pixel)

    def set_pixel(self, x: int, y: int, color: ColorType) -> None:
        """设置指定位置的像素颜色值
        
        Args:
            x: 像素x坐标(水平方向)
            y: 像素y坐标(垂直方向)
            color: 颜色值(格式需与当前模式匹配)
            
        Raises:
            IndexError: 当坐标超出矩阵范围时抛出
            TypeError: 颜色类型与模式不匹配时抛出
            ValueError: 颜色值超出有效范围时抛出
        """
        # 验证坐标有效性
        self.__is_coordinate_out_of_range_ex(x, y)
        
        # 验证颜色值合法性
        self.__validate_color_by_mode(color)
        
        # 根据模式设置像素值
        if self.mode == ColorMode.MONO:
            self.__matrix[y, x] = bool(color)
        elif self.mode in [ColorMode.GRAY_8, ColorMode.RGB565]:
            self.__matrix[y, x] = int(color)
        elif self.mode in [ColorMode.RGB_888, ColorMode.RGB_8888]:
            self.__matrix[y, x] = color

    def is_coordinate_out_of_range(self, x: int, y: int) -> bool:
        """判断坐标是否超出矩阵范围
        
        Args:
            x: 待检查的x坐标
            y: 待检查的y坐标
            
        Returns:
            超出范围返回True,否则返回False
        """
        return not (0 <= x < self.__width and 0 <= y < self.__height)

    def __is_coordinate_out_of_range_ex(self, x: int, y: int) -> bool:
        """验证坐标有效性,超出范围则抛出异常
        
        Args:
            x: 待检查的x坐标
            y: 待检查的y坐标
            
        Returns:
            坐标有效时返回True
            
        Raises:
            IndexError: 当坐标超出矩阵范围时抛出
        """
        if self.is_coordinate_out_of_range(x, y):
            raise IndexError(
                f"坐标 ({x}, {y}) 超出范围(宽: {self.__width}, 高: {self.__height})"
            )
        return True

    def __validate_color_by_mode(self, color: ColorType) -> None:
        """根据当前模式验证颜色值的合法性
        
        Args:
            color: 待验证的颜色值
            
        Raises:
            TypeError: 颜色类型与模式不匹配
            ValueError: 颜色值超出有效范围
        """
        if self.mode == ColorMode.MONO:
            if not isinstance(color, (int, bool)):
                raise TypeError(f"二值模式需要int或bool类型,实际为{type(color)}")
            if isinstance(color, int) and color not in (0, 1):
                raise ValueError(f"二值模式整数只能是0或1,实际为{color}")
        
        elif self.mode == ColorMode.GRAY_8:
            if not isinstance(color, int):
                raise TypeError(f"灰度模式需要int类型,实际为{type(color)}")
            if not (0 <= color <= 255):
                raise ValueError(f"灰度模式值需在0-255之间,实际为{color}")
        
        elif self.mode == ColorMode.RGB_888:
            if not isinstance(color, (tuple, list)):
                raise TypeError(f"RGB888模式需要元组或列表,实际为{type(color)}")
            if len(color) != 3:
                raise ValueError(f"RGB888模式需要3个通道值,实际为{len(color)}个")
            for i, val in enumerate(color):
                if not isinstance(val, int) or not (0 <= val <= 255):
                    raise ValueError(f"RGB888第{i+1}通道需为0-255整数,实际为{val}")
        
        elif self.mode == ColorMode.RGB_8888:
            if not isinstance(color, (tuple, list)):
                raise TypeError(f"RGB8888模式需要元组或列表,实际为{type(color)}")
            if len(color) != 4:
                raise ValueError(f"RGB8888模式需要4个通道值,实际为{len(color)}个")
            for i, val in enumerate(color):
                if not isinstance(val, int) or not (0 <= val <= 255):
                    raise ValueError(f"RGB8888第{i+1}通道需为0-255整数,实际为{val}")
        
        elif self.mode == ColorMode.RGB565:
            if not isinstance(color, int):
                raise TypeError(f"RGB565模式需要int类型,实际为{type(color)}")
            if not (0 <= color <= 0xFFFF):
                raise ValueError(f"RGB565模式值需在0-65535之间,实际为{color}")
            
    def fill(self, color: ColorType) -> None:
        """用指定颜色填充整个矩阵(比循环调用 set_pixel 高效)
        
        Args:
            color: 填充颜色值(格式需与当前模式匹配)
            
        Raises:
            TypeError: 颜色类型与模式不匹配
            ValueError: 颜色值超出有效范围
        """
        self.__validate_color_by_mode(color)
        if self.mode in [ColorMode.MONO, ColorMode.GRAY_8, ColorMode.RGB565]:
            # 单值模式直接填充标量
            self.__matrix.fill(color)
        else:
            # 多通道模式填充颜色元组(需转换为numpy数组格式)
            self.__matrix[:] = np.array(color, dtype=self.__matrix.dtype)
            
    def get_region(self, x1: int, y1: int, x2: int, y2: int) -> "PixelMatrix":
        """提取矩形区域的子矩阵(左闭右开区间 [x1, x2) x [y1, y2))
        
        Args:
            x1: 区域左上角x坐标
            y1: 区域左上角y坐标
            x2: 区域右下角x坐标(不含)
            y2: 区域右下角y坐标(不含)
            
        Returns:
            包含指定区域的新PixelMatrix实例
            
        Raises:
            IndexError: 当区域坐标超出矩阵范围时抛出
        """
        # 校验区域坐标有效性
        for x in [x1, x2]:
            for y in [y1, y2]:
                self.__is_coordinate_out_of_range_ex(x, y)
        
        # 切片获取子区域并创建新实例
        region_data = self.__matrix[y1:y2, x1:x2].copy()
        return PixelMatrix(region_data, self.mode)

    def set_region(self, x: int, y: int, sub_matrix: "PixelMatrix") -> None:
        """将子矩阵覆盖到当前矩阵的指定位置
        
        Args:
            x: 目标位置左上角x坐标
            y: 目标位置左上角y坐标
            sub_matrix: 要覆盖的子矩阵
            
        Raises:
            ValueError: 子矩阵模式与当前模式不匹配时抛出
            IndexError: 子矩阵超出当前矩阵范围时抛出
        """
        if sub_matrix.mode != self.mode:
            raise ValueError(f"子矩阵模式 {sub_matrix.mode} 与当前模式 {self.mode} 不匹配")
        
        # 校验目标位置是否足够容纳子矩阵
        self.__is_coordinate_out_of_range_ex(
            x + sub_matrix.width - 1, 
            y + sub_matrix.height - 1
        )
        
        # 批量覆盖数据
        self.__matrix[y:y+sub_matrix.height, x:x+sub_matrix.width] = sub_matrix.matrix
    

    @staticmethod
    def from_image(image: Image.Image, mode: Optional[ColorMode] = None) -> "PixelMatrix":
        """从PIL Image对象创建PixelMatrix(支持自动模式推导和RGB565转换)
        
        Args:
            image: PIL Image对象
            mode: 目标颜色模式,若为None则自动推导
            
        Returns:
            新的PixelMatrix实例
        """
        # 自动推导目标模式(若未指定)
        if mode is None:
            mode_map = {
                "1": ColorMode.MONO,    # PIL的"1"模式对应二值图像
                "L": ColorMode.GRAY_8,  # PIL的"L"模式对应8位灰度
                "RGB": ColorMode.RGB_888,  # PIL的"RGB"对应24位RGB
                "RGBA": ColorMode.RGB_8888 # PIL的"RGBA"对应32位RGBA
            }
            mode = mode_map.get(image.mode, ColorMode.RGB_888)  # 默认转为RGB888
        
        # 处理RGB565模式: 需从其他模式手动转换
        if mode == ColorMode.RGB565:
            # 先将图像转为RGB模式(确保三通道数据)
            rgb_image = image.convert("RGB")
            # 转换为numpy数组(RGB888格式)
            rgb_data = np.array(rgb_image, dtype=np.uint8)
            # 提取R、G、B通道并压缩为RGB565格式
            r = (rgb_data[..., 0] >> 3) & 0x1F  # 取高5位
            g = (rgb_data[..., 1] >> 2) & 0x3F  # 取高6位
            b = (rgb_data[..., 2] >> 3) & 0x1F  # 取高5位
            # 组合为16位整数(R<<11 | G<<5 | B)
            rgb565_data = (r << 11) | (g << 5) | b
            # 创建RGB565模式的PixelMatrix
            return PixelMatrix(rgb565_data.astype(np.uint16), ColorMode.RGB565)
        
        # 处理其他模式: 直接转换为对应格式
        # 先将PIL图像转为目标模式对应的PIL模式(确保数据兼容)
        pil_mode_map = {
            ColorMode.MONO: "1",
            ColorMode.GRAY_8: "L",
            ColorMode.RGB_888: "RGB",
            ColorMode.RGB_8888: "RGBA"
        }
        target_pil_mode = pil_mode_map[mode]
        converted_image = image.convert(target_pil_mode)
        # 转换为numpy数组并创建实例
        image_data = np.array(converted_image, dtype=_VALID_MODES[mode]["dtype"])
        return PixelMatrix(image_data, mode)
    
    def to_image(self) -> Image.Image:
        """转换为PIL Image对象(便于预览、保存)
        
        Returns:
            对应的PIL Image对象
        """
        mode_map = {
            ColorMode.MONO: "1", 
            ColorMode.GRAY_8: "L", 
            ColorMode.RGB_888: "RGB", 
            ColorMode.RGB_8888: "RGBA",
            ColorMode.RGB565: "RGB"  # RGB565需先转为RGB888才能显示
        }
        pil_mode = mode_map[self.mode]
        
        # 特殊处理RGB565转RGB888(需解码16位数据)
        if self.mode == ColorMode.RGB565:
            rgb_data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            rgb_data[..., 0] = ((self.__matrix >> 11) & 0x1F) << 3  # R5→R8
            rgb_data[..., 1] = ((self.__matrix >> 5) & 0x3F) << 2   # G6→G8
            rgb_data[..., 2] = (self.__matrix & 0x1F) << 3          # B5→B8
            return Image.fromarray(rgb_data, mode=pil_mode)
            
        return Image.fromarray(self.__matrix, mode=pil_mode)
    
    def to_numpy(self) -> np.ndarray:
        """返回底层Numpy数组的视图(不复制数据)
        
        Returns:
            底层存储的numpy数组
        """
        return self.__matrix
    
    def to_bytes(self) -> bytes:
        """将像素数据转换为字节流(用于硬件传输)
        
        Raises:
            NotImplementedError: 方法尚未实现
        """
        raise NotImplementedError("尚未实现PixelMatrix的to_bytes方法")
    
    def convert_mode(self, mode: ColorMode) -> "PixelMatrix":
        """转换为指定的颜色模式
        
        Args:
            mode: 目标颜色模式
            
        Returns:
            转换后的新PixelMatrix实例
        """
        #  TODO: 实现不同模式间的转换逻辑
        return PixelMatrix(self.matrix, mode)
    
    def copy(self) -> "PixelMatrix":
        """创建当前像素矩阵的深拷贝
        
        Returns:
            新的PixelMatrix实例,包含当前矩阵的副本
        """
        return PixelMatrix(self.matrix, self.mode)
    
    def __repr__(self) -> str:
        return f"<PixelMatrix (mode={self.mode}, size=({self.width}x{self.height})>"
    
    def __str__(self) -> str:
        return repr(self)


# ------------------------------
# 帧类: 包含像素矩阵和时间信息
# ------------------------------
class Frame:
    """帧容器,包含一帧图像的像素数据、时间戳和元数据
    
    用于视频序列或动画中,表示在特定时间点显示的一帧图像,
    封装了PixelMatrix并提供额外的时间相关属性。
    """
    __pixel_matrix: PixelMatrix
    __timestamp: float  # 帧的时间戳(秒)
    __metadata: Dict[str, Any]  # 帧的额外信息
    __frame_index: int  # 帧在序列中的索引

    def __init__(
        self, 
        pixel_matrix: PixelMatrix, 
        timestamp: float = 0.0, 
        frame_index: int = 0, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """初始化帧对象
        
        Args:
            pixel_matrix: 帧包含的像素矩阵
            timestamp: 帧的时间戳(秒),默认为0.0
            frame_index: 帧在序列中的索引,默认为0
            metadata: 帧的额外信息,默认为空字典
        """
        self.__pixel_matrix = pixel_matrix
        self.__timestamp = timestamp
        self.__metadata = metadata.copy() if metadata else {}
        self.__frame_index = frame_index
        
    @staticmethod
    def from_image(
        image: Image.Image, 
        timestamp: float = 0.0, 
        frame_index: int = 0
    ) -> "Frame":
        """从PIL Image对象创建Frame
        
        Args:
            image: PIL Image对象
            timestamp: 帧的时间戳(秒),默认为0.0
            frame_index: 帧在序列中的索引,默认为0
            
        Returns:
            新的Frame实例
        """
        return Frame(PixelMatrix.from_image(image), timestamp, frame_index)
    
    @staticmethod
    def from_pixel_matrix(
        matrix: PixelMatrix, 
        timestamp: float = 0.0, 
        frame_index: int = 0
    ) -> "Frame":
        """从PixelMatrix对象创建Frame
        
        Args:
            matrix: PixelMatrix对象
            timestamp: 帧的时间戳(秒),默认为0.0
            frame_index: 帧在序列中的索引,默认为0
            
        Returns:
            新的Frame实例
        """
        return Frame(matrix, timestamp, frame_index)
    
    @property
    def pixel_matrix(self) -> np.ndarray:
        """返回像素矩阵的副本(避免外部直接修改)"""
        return self.__pixel_matrix.matrix

    @property
    def timestamp(self) -> float:
        """返回帧的时间戳(秒)"""
        return self.__timestamp
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """返回帧元数据的深拷贝(避免外部直接修改)"""
        return copy.deepcopy(self.__metadata)
    
    @property
    def frame_index(self) -> int:
        """返回帧在序列中的索引"""
        return self.__frame_index
    
    @property
    def width(self) -> int:
        """返回帧的宽度(像素)"""
        return self.__pixel_matrix.width
    
    @property
    def height(self) -> int:
        """返回帧的高度(像素)"""
        return self.__pixel_matrix.height
    
    @property
    def mode(self) -> ColorMode:
        """返回帧的颜色模式"""
        return self.__pixel_matrix.mode
    
    def get_pixel(self, x: int, y: int) -> ColorType:
        """获取指定坐标的像素值
        
        Args:
            x: 像素x坐标(水平方向)
            y: 像素y坐标(垂直方向)
            
        Returns:
            像素值(格式与当前模式匹配)
        """
        return self.__pixel_matrix.get_pixel(x, y)
    
    def set_pixel(self, x: int, y: int, color: ColorType) -> None:
        """设置指定位置的像素颜色值
        
        Args:
            x: 像素x坐标(水平方向)
            y: 像素y坐标(垂直方向)
            color: 颜色值(格式需与当前模式匹配)
        """
        self.__pixel_matrix.set_pixel(x, y, color)
    
    def fill(self, color: ColorType) -> None:
        """用指定颜色填充整个帧
        
        Args:
            color: 填充颜色值(格式需与当前模式匹配)
        """
        self.__pixel_matrix.fill(color)
        
    def get_region(self, x1: int, y1: int, x2: int, y2: int) -> "Frame":
        """提取矩形区域作为新的帧
        
        Args:
            x1: 区域左上角x坐标
            y1: 区域左上角y坐标
            x2: 区域右下角x坐标(不含)
            y2: 区域右下角y坐标(不含)
            
        Returns:
            包含指定区域的新Frame实例
        """
        sub_matrix = self.__pixel_matrix.get_region(x1, y1, x2, y2)
        return Frame.from_pixel_matrix(sub_matrix, self.timestamp, self.frame_index)
    
    def to_image(self) -> Image.Image:
        """转换为PIL Image对象
        
        Returns:
            对应的PIL Image对象
        """
        return self.__pixel_matrix.to_image()
    
    def to_pixel_matrix(self) -> PixelMatrix:
        """转换为PixelMatrix对象(返回副本)
        
        Returns:
            新的PixelMatrix实例,包含当前帧的像素数据
        """
        return self.__pixel_matrix.copy()
    
    def to_bytes(self) -> bytes:
        """将帧数据转换为字节流
        
        Returns:
            帧数据的字节表示
        """
        return self.__pixel_matrix.to_bytes()
    
    def convert_mode(self, mode: ColorMode) -> "Frame":
        """转换为指定的颜色模式
        
        Args:
            mode: 目标颜色模式
            
        Returns:
            转换后的新Frame实例
        """
        return Frame.from_pixel_matrix(
            self.__pixel_matrix.convert_mode(mode),
            self.timestamp,
            self.frame_index
        )
    
    def is_coordinate_out_of_range(self, x: int, y: int) -> bool:
        """判断坐标是否超出帧范围
        
        Args:
            x: 待检查的x坐标
            y: 待检查的y坐标
            
        Returns:
            超出范围返回True,否则返回False
        """
        return self.__pixel_matrix.is_coordinate_out_of_range(x, y)
    
    def copy(self) -> "Frame":
        """创建当前帧的深拷贝
        
        Returns:
            新的Frame实例,包含当前帧的所有数据副本
        """
        return Frame(
            self.__pixel_matrix.copy(),
            self.__timestamp,
            self.frame_index,
            self.metadata
        )
    
    def to_numpy(self) -> np.ndarray:
        """返回底层Numpy数组
        
        Returns:
            帧像素数据的numpy数组
        """
        return self.__pixel_matrix.to_numpy()
    
    def __repr__(self) -> str:
        return (f"<Frame (mode={self.mode}, size=({self.width}x{self.height}), "
                f"frame_index={self.__frame_index}, timestamp={self.timestamp:.3f})>")
        
    def __str__(self) -> str:
        return repr(self)


# ------------------------------
# 屏幕配置类: 封装硬件相关参数
# ------------------------------
class ScreenConfig:
    """屏幕配置数据类,封装所有硬件相关参数
    
    该类定义了显示设备的硬件特性,包括分辨率、支持的颜色模式、
    扫描方向等,用于确保图像数据与硬件兼容。
    """
    
    # ------------------------------
    # 核心字段(硬件标识与基础参数)
    # ------------------------------
    width: int  # 屏幕宽度(像素)
    height: int  # 屏幕高度(像素)
    screen_type: ScreenType  # 屏幕硬件类型
    
    # ------------------------------
    # 扫描与数据传输相关字段
    # ------------------------------
    scan_direction: ScanDirection = ScanDirection.HORIZONTAL  # 扫描方向
    bit_order: BitOrder = BitOrder.MSB_FIRST  # 位序
    row_offset: int = 0  # 行偏移量(硬件校准用)
    col_offset: int = 0  # 列偏移量(硬件校准用)
    
    # ------------------------------
    # 扩展元数据
    # ------------------------------
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外硬件信息
    
    
    def __init__(self, width, height,
                scan_direction: ScanDirection=ScanDirection.HORIZONTAL,
                bit_order: BitOrder=BitOrder.MSB_FIRST,
                row_offset: int=0,
                col_offset: int=0,
                metadata: Dict[str, Any] = dict()
            ):
        self.width = width
        self.height = height
        self.scan_direction = scan_direction
        self.bit_order = bit_order
        self.row_offset = row_offset
        self.col_offset = col_offset
        self.metadata = metadata
        
    
    
    # ------------------------------
    # 初始化校验方法
    # ------------------------------
    def __post_init__(self):
        """初始化后自动校验参数合法性"""
        self._validate_resolution()
    
    def _validate_resolution(self) -> None:
        """校验分辨率为正整数
        
        Raises:
            ValueError: 当分辨率为非正整数时抛出
        """
        if self.width <= 0 or self.height <= 0 or type(self.width) != int or type(self.height) != int:
            raise ValueError(f"无效分辨率: 宽={self.width}, 高={self.height}(必须为正整数)")
    
    def _validate_color_mode(self) -> None:
        """校验当前色彩模式在支持列表中
        
        Raises:
            ValueError: 当前模式不在支持列表中时抛出
        """
        if self.color_mode not in self.supported_modes:
            supported = [str(m) for m in self.supported_modes]
            raise ValueError(
                f"不支持的色彩模式: {self.color_mode},支持模式: {supported}"
            )
    
    # ------------------------------
    # 核心属性访问方法
    # ------------------------------
    @property
    def resolution(self) -> Tuple[int, int]:
        """返回分辨率元组 (宽, 高)"""
        return (self.width, self.height)
    
    @property
    def pixel_count(self) -> int:
        """返回总像素数"""
        return self.width * self.height
    
    
    # ------------------------------
    # 功能校验方法
    # ------------------------------
    def supports_mode(self, mode: ColorMode) -> bool:
        """检查屏幕是否支持指定色彩模式
        
        Args:
            mode: 待检查的颜色模式
            
        Returns:
            支持则返回True,否则返回False
        """
        return mode in self.supported_modes
    
    def is_compatible_with(self, matrix: PixelMatrix) -> bool:
        """检查像素矩阵是否与当前屏幕配置兼容(分辨率+色彩模式)
        
        Args:
            matrix: 待检查的PixelMatrix对象
            
        Returns:
            兼容则返回True,否则返回False
        """
        return (matrix.width == self.width 
                and matrix.height == self.height 
                and self.supports_mode(matrix.mode))
    
    
    # ------------------------------
    # 序列化方法
    # ------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典,用于配置存储或传输
        
        Returns:
            包含所有配置信息的字典
        """
        return {
            "screen_type": str(self.screen_type),
            "color_mode": str(self.color_mode),
            "supported_modes": [str(m) for m in self.supported_modes],
            "scan_direction": str(self.scan_direction),
            "bit_order": str(self.bit_order),
            "row_offset": self.row_offset,
            "col_offset": self.col_offset,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScreenConfig":
        """从字典创建实例,用于配置加载
        
        Args:
            data: 包含配置信息的字典
            
        Returns:
            新的ScreenConfig实例
        """
        # 转换字符串为枚举类型
        screen_type = ScreenType[data["screen_type"].upper()]
        color_mode = ColorMode[data["color_mode"].upper()]
        supported_modes = [
            ColorMode[m.upper()] for m in data["supported_modes"]
        ]
        scan_direction = ScanDirection[data.get("scan_direction", "horizontal").upper()]
        bit_order = BitOrder[data.get("bit_order", "msb_first").upper()]
        
        return cls(
            width=data["resolution"][0],
            height=data["resolution"][1],
            screen_type=screen_type,
            color_mode=color_mode,
            supported_modes=supported_modes,
            scan_direction=scan_direction,
            bit_order=bit_order,
            row_offset=data.get("row_offset", 0),
            col_offset=data.get("col_offset", 0),
            metadata=data.get("metadata", {})
        )
