from typing import Iterable, Tuple, List

from data_structures import ScreenConfig, PixelMatrix, validate_mode, ColorMode, ScanDirection, BitOrder

class ScanStrategy:
    """
    扫描策略基类:定义像素遍历顺序的抽象接口
    
    不同屏幕硬件的像素扫描方向不同(如行扫、列扫、反向扫描等),
    通过子类实现具体的扫描逻辑,实现扫描方式的灵活扩展。
    """
    
    __width: int  # 屏幕宽度(像素)
    __height: int  # 屏幕高度(像素)
    
    def __init__(self, width: int, height: int) -> None:
        """
        初始化扫描策略
        
        Args:
            width: 屏幕宽度(像素)
            height: 屏幕高度(像素)
        """
        self.__width = width
        self.__height = height
    
    def get_pixel_order(self) -> Iterable[Tuple[int, int]]:
        """
        抽象方法:返回像素坐标的遍历顺序迭代器
        
        子类需实现具体的扫描逻辑,返回 (x, y) 坐标的迭代序列,
        决定像素数据打包为字节流的顺序。
        
        Returns:
            像素坐标迭代器,每个元素为 (x, y) 元组
        """
        raise NotImplementedError("子类必须实现 get_pixel_order 方法")


class HorizontalScanStrategy(ScanStrategy):
    """水平扫描策略:按行优先遍历像素(从左到右,从上到下)"""
    
    def __init__(self, width: int, height: int) -> None:
        """
        初始化水平扫描策略
        
        Args:
            width: 屏幕宽度(像素)
            height: 屏幕高度(像素)
        """
        super().__init__(width, height)
    
    def get_pixel_order(self) -> Iterable[Tuple[int, int]]:
        """
        生成水平扫描的像素坐标序列
        
        遍历顺序:
        - 行从 0 到 height-1(从上到下)
        - 每行内从 x=0 到 x=width-1(从左到右)
        
        Returns:
            水平扫描顺序的 (x, y) 坐标迭代器
        """
        for y in range(self._ScanStrategy__height):
            for x in range(self._ScanStrategy__width):
                yield (x, y)


class HorizontalReversedScanStrategy(ScanStrategy):
    """水平反向扫描策略:按行优先遍历,但每行从右到左"""
    
    def __init__(self, width: int, height: int) -> None:
        """
        初始化水平反向扫描策略
        
        Args:
            width: 屏幕宽度(像素)
            height: 屏幕高度(像素)
        """
        super().__init__(width, height)
    
    def get_pixel_order(self) -> Iterable[Tuple[int, int]]:
        """
        生成水平反向扫描的像素坐标序列
        
        遍历顺序:
        - 行从 0 到 height-1(从上到下)
        - 每行内从 x=width-1 到 x=0(从右到左)
        
        Returns:
            水平反向扫描顺序的 (x, y) 坐标迭代器
        """
        for y in range(self._ScanStrategy__height):
            for x in range(self._ScanStrategy__width - 1, -1, -1):
                yield (x, y)


class VerticalScanStrategy(ScanStrategy):
    """垂直扫描策略:按列优先遍历像素(从上到下,从左到右)"""
    
    def __init__(self, width: int, height: int) -> None:
        """
        初始化垂直扫描策略
        
        Args:
            width: 屏幕宽度(像素)
            height: 屏幕高度(像素)
        """
        super().__init__(width, height)
    
    def get_pixel_order(self) -> Iterable[Tuple[int, int]]:
        """
        生成垂直扫描的像素坐标序列
        
        遍历顺序:
        - 列从 x=0 到 x=width-1(从左到右)
        - 每列内从 y=0 到 y=height-1(从上到下)
        
        Returns:
            垂直扫描顺序的 (x, y) 坐标迭代器
        """
        for x in range(self._ScanStrategy__width):
            for y in range(self._ScanStrategy__height):
                yield (x, y)


class VerticalReversedScanStrategy(ScanStrategy):
    """垂直反向扫描策略:按列优先遍历,但每列从下到上"""
    
    def __init__(self, width: int, height: int) -> None:
        """
        初始化垂直反向扫描策略
        
        Args:
            width: 屏幕宽度(像素)
            height: 屏幕高度(像素)
        """
        super().__init__(width, height)
    
    def get_pixel_order(self) -> Iterable[Tuple[int, int]]:
        """
        生成垂直反向扫描的像素坐标序列
        
        遍历顺序:
        - 列从 x=0 到 x=width-1(从左到右)
        - 每列内从 y=height-1 到 y=0(从下到上)
        
        Returns:
            垂直反向扫描顺序的 (x, y) 坐标迭代器
        """
        for x in range(self._ScanStrategy__width):
            for y in range(self._ScanStrategy__height - 1, -1, -1):
                yield (x, y)


class VerticalScanStrategyByPage(ScanStrategy):
    """按页垂直扫描策略:按列优先遍历"""
    
    def __init__(self, width: int, height: int) -> None:
        """
        初始化垂直反向扫描策略
        
        Args:
            width: 屏幕宽度(像素)
            height: 屏幕高度(像素)
        """
        super().__init__(width, height)
    
    def get_pixel_order(self) -> Iterable[Tuple[int, int]]:
        if self._ScanStrategy__height % 8:
            raise Exception("屏幕高度不是8的倍数, 不能按Page垂直扫描")
        for pageOrder in range(self._ScanStrategy__height // 8):
            for x in range(self._ScanStrategy__width):
                for y in range(8):
                    yield (x, pageOrder * 8 + y)
                    
class VerticalReversedScanStrategyByPage(ScanStrategy):
    """按页垂直反向扫描策略:按列优先遍历,但每列从下到上"""
    
    def __init__(self, width: int, height: int) -> None:
        """
        初始化垂直反向扫描策略
        
        Args:
            width: 屏幕宽度(像素)
            height: 屏幕高度(像素)
        """
        super().__init__(width, height)
    
    def get_pixel_order(self) -> Iterable[Tuple[int, int]]:
        if self._ScanStrategy__height % 8:
            raise Exception("屏幕高度不是8的倍数, 不能按Page垂直扫描")
        for pageOrder in range(self._ScanStrategy__height // 8):
            for x in range(self._ScanStrategy__width):
                for y in range(7, -1, -1):
                    yield (x, pageOrder * 8 + y)
                    

class BitEncodingStrategy:
    """
    位编码策略基类:定义像素数据到位的编码逻辑
    
    不同硬件对字节内的位排列顺序要求不同(高位优先/低位优先),
    通过子类实现具体的编码方式,适配不同硬件的位序要求。
    """
    
    def encode_by_bits(self, pixels: List[int]) -> int:
        """
        抽象方法:将8个像素编码为1个字节
        
        输入的像素列表长度固定为8,每个元素为0(熄灭)或1(点亮),
        子类需实现如何将这8个像素映射到字节的8个比特位。
        
        Args:
            pixels: 长度为8的像素列表,元素为0或1
            
        Returns:
            编码后的字节值(0-255的整数)
        """
        raise NotImplementedError("子类必须实现 encode 方法")
        
    def pad_remaining(self, pixels: List[int]) -> List[int]:
        """
        填充不足8个的像素列表,使其长度为8
        
        当像素总数不是8的倍数时,用0填充剩余位置,确保能完整编码为一个字节。
        
        Args:
            pixels: 长度小于8的像素列表(元素为0或1)
            
        Returns:
            填充后的像素列表(长度为8)
        """
        if len(pixels) > 8:
            raise ValueError(f"像素列表长度不能超过8,当前为{len(pixels)}")
        return pixels + [0] * (8 - len(pixels))
    
    def break_decimal_into_binary(self, decimal: int) -> List[int]:
        bits = []
        for i in range(8):
            bits.append((decimal & (0x80 >> i)) >> (7 - i))
        return bits

    def encoding_by_decimal(self, decimal: int) -> int:
        return self.encode_by_bits(self.break_decimal_into_binary(decimal))

class MSBFirstEncoding(BitEncodingStrategy):
    """
    高位优先编码策略:第一个像素对应字节的最高位(bit7)
    
    其实如果是这种策略, 那对应字节可以不被这个类处理, 直接跳过
    """
    
    def encode_by_bits(self, pixels: List[int]) -> int:
        """
        将8个像素按高位优先编码为字节
        
        编码规则:
        - pixels[0] → bit7(最高位)
        - pixels[1] → bit6
        - ...
        - pixels[7] → bit0(最低位)
        
        示例:
        pixels = [1,0,1,1,0,0,1,0] → 0b10110010 → 178(十进制)
        
        Args:
            pixels: 长度为8的像素列表(元素为0或1)
            
        Returns:
            编码后的字节值(0-255)
        """
        if len(pixels) != 8:
            raise ValueError(f"需8个像素进行编码,当前为{len(pixels)}个")
        
        byte_value = 0
        for i in range(8):
            # 像素值左移 (7 - i) 位,对应到字节的高位到低位
            byte_value |= pixels[i] << (7 - i)
        return byte_value


class LSBFirstEncoding(BitEncodingStrategy):
    """低位优先编码策略:第一个像素对应字节的最低位(bit0)"""
    
    def encode_by_bits(self, pixels: List[int]) -> int:
        """
        将8个像素按低位优先编码为字节
        
        编码规则:
        - pixels[0] → bit0(最低位)
        - pixels[1] → bit1
        - ...
        - pixels[7] → bit7(最高位)
        
        示例:
        pixels = [1,0,1,1,0,0,1,0] → 0b01001101 → 77(十进制)
        
        Args:
            pixels: 长度为8的像素列表(元素为0或1)
            
        Returns:
            编码后的字节值(0-255)
        """
        if len(pixels) != 8:
            raise ValueError(f"需8个像素进行编码,当前为{len(pixels)}个")
        
        byte_value = 0
        for i in range(8):
            # 像素值左移 i 位,对应到字节的低位到高位
            byte_value |= pixels[i] << i
        return byte_value


def find_scan_strategy(config: ScreenConfig) -> ScanStrategy:
    """
    根据屏幕配置创建对应的扫描策略类
    
    映射关系:
    - "horizontal" → HorizontalScanStrategy
    - "horizontal_reversed" → HorizontalReversedScanStrategy
    - "vertical" → VerticalScanStrategy
    - "vertical_reversed" → VerticalReversedScanStrategy
    
    Args:
        config: 屏幕配置对象,包含扫描方向信息
        
    Returns:
        扫描策略实例
        
    Raises:
        ValueError: 不支持的扫描方向
    """
    strategy_map = {
        ScanDirection.HORIZONTAL: HorizontalScanStrategy,
        ScanDirection.HORIZONTAL_REVERSED: HorizontalReversedScanStrategy,
        ScanDirection.VERTICAL: VerticalScanStrategy,
        ScanDirection.VERTICAL_REVERSED: VerticalReversedScanStrategy,
        ScanDirection.VERTICAL_BY_PAGE: VerticalScanStrategyByPage,
        ScanDirection.VERTICAL_REVERSED_BY_PAGE: VerticalReversedScanStrategyByPage
    }
    
    strategy_cls = strategy_map.get(config.scan_direction)
    if not strategy_cls:
        raise ValueError(
            f"不支持的扫描方向:{config.scan_direction},"
            f"支持的方向:{list(strategy_map.keys())}"
        )
    
    return strategy_cls


def find_bit_encoding_strategy(config: ScreenConfig) -> BitEncodingStrategy:
    """
    根据屏幕配置创建对应的位编码策略类
    
    映射关系:
    - "msb_first" → MSBFirstEncoding
    - "lsb_first" → LSBFirstEncoding
    
    Args:
        config: 屏幕配置对象,包含位序信息
        
    Returns:
        位编码策略实例
        
    Raises:
        ValueError: 不支持的位序
    """
    if config.bit_order == BitOrder.MSB_FIRST:
        return MSBFirstEncoding
    elif config.bit_order == BitOrder.LSB_FIRST:
        return LSBFirstEncoding
    else:
        raise ValueError(
            f"不支持的位序:{config.bit_order},支持的位序:msb_first, lsb_first"
        )


def pack_pixels(pixels: Iterable[int], encoding: BitEncodingStrategy) -> bytes:
    """
    将像素序列按编码策略打包为字节流
    
    流程:
    1. 迭代像素序列,每8个像素为一组
    2. 不足8个的组用0填充
    3. 每组像素通过编码策略转换为1个字节
    4. 所有字节拼接为最终字节流
    
    Args:
        pixels: 像素迭代器(元素为0或1)
        encoding: 位编码策略实例
        
    Returns:
        打包后的字节流
    """
    byte_list = []
    current_group = []
    
    for pixel in pixels:
        # 确保像素值为0或1(单色模式)
        if pixel not in (0, 1):
            raise ValueError(f"单色模式像素值必须为0或1,当前为{pixel}")
        
        current_group.append(pixel)
        
        # 每8个像素编码为1个字节
        if len(current_group) == 8:
            byte_value = encoding.encode(current_group)
            byte_list.append(byte_value)
            current_group = []
    
    # 处理剩余不足8个的像素
    if current_group:
        padded_group = encoding.pad_remaining(current_group)
        byte_value = encoding.encode(padded_group)
        byte_list.append(byte_value)
    
    # 转换为bytes对象返回
    return bytes(byte_list)


class ModulateStrategy:
    def __init__(self, matrix: PixelMatrix, scan_strategy: ScanStrategy, encoding_strategy: BitEncodingStrategy):
        self.matrix = matrix
        self.scan_strategy:ScanStrategy = scan_strategy(self.matrix.width, self.matrix.height)
        self.encoding_strategy: BitEncodingStrategy = encoding_strategy()
    
    def modulate(self) -> bytes:
        raise NotImplementedError("ModulateStrategy是一个抽象基类, 不提供任何实现")
        
class BinaryModulateStrategy(ModulateStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def modulate(self) -> bytes:
        coordinates = self.scan_strategy.get_pixel_order()
        return bytes(self.encoding_strategy.encode_by_bits([self.matrix.get_pixel(x, y)[0] for bytes in coordinates]))
    
class GreyModulateStrategy(ModulateStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def modulate(self) -> bytes:
        coordinates = self.scan_strategy.get_pixel_order()
        if type(self.encoding_strategy) == MSBFirstEncoding:
            return bytes(self.matrix.get_pixel(x, y)[0] for x, y in coordinates)
        return bytes(self.encoding_strategy.encoding_by_decimal(self.matrix.get_pixel(x, y)[0]) for x, y in coordinates)
        

class RgbModulateStrategy(ModulateStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def modulate(self) -> bytes:
        coordinates = self.scan_strategy.get_pixel_order()
        bs = []
        if type(self.encoding_strategy) == MSBFirstEncoding:
            for x, y in coordinates:
                for val in self.matrix.get_pixel(x, y):
                    bs.append(val)
        else:
            for x, y in coordinates:
                for val in self.matrix.get_pixel(x, y):
                    bs.append(self.encoding_strategy.encoding_by_decimal(val))
        return bytes(bs)
                    
class Rgb565ModulateStrategy(RgbModulateStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Rgb888ModulateStrategy(RgbModulateStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Rgb8888ModulateStrategy(RgbModulateStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

MODULATE_STRATEGY_MAPPING = {
    ColorMode.MONO: BinaryModulateStrategy,
    ColorMode.GRAY_8: GreyModulateStrategy,
    ColorMode.RGB565: Rgb565ModulateStrategy,
    ColorMode.RGB_888: Rgb888ModulateStrategy,
    ColorMode.RGB_8888: Rgb8888ModulateStrategy
}

class Modulator:
    """
    取模引擎主类:负责将 PixelMatrix 按 ScreenConfig 转换为硬件可直接驱动的字节流
    
    核心功能:
    - 校验输入像素矩阵与屏幕配置的兼容性
    - 根据屏幕配置选择合适的扫描策略和位编码策略
    - 协调各组件完成像素数据到硬件字节流的转换
    """
    __screen_config: ScreenConfig  # 屏幕配置参数(包含扫描方向、位序等硬件信息)
    
    def __init__(self, screen_config: ScreenConfig) -> None:
        """
        初始化取模引擎
        
        Args:
            screen_config: 屏幕配置对象,包含硬件相关的所有参数
        """
        self.__screen_config = screen_config
        
    def modulate(self, matrix: PixelMatrix) -> bytes:
        """
        核心取模方法:将像素矩阵转换为硬件可识别的字节流
        
        流程:
        1. 校验输入矩阵的合法性(模式、分辨率)
        2. 根据屏幕配置创建扫描策略和位编码策略
        3. 按扫描策略获取像素遍历顺序
        4. 提取像素值并按位编码策略打包为字节流
        
        Args:
            matrix: 待转换的像素矩阵(必须为单色模式)
            
        Returns:
            编码后的字节流,可直接发送给屏幕硬件
            
        Raises:
            ValueError: 矩阵与屏幕配置不兼容(模式或分辨率不匹配)
        """
        # 验证输入合法性
        self.__validate_input(matrix)
        
        # 找到扫描策略和编码策略类
        scan_strategy = find_scan_strategy(self.__screen_config)
        bit_encoder = find_bit_encoding_strategy(self.__screen_config)
        
        # 创建取模策略对象的实例
        modulate_strategy:ModulateStrategy = MODULATE_STRATEGY_MAPPING[matrix.mode](matrix, scan_strategy, bit_encoder)
        return modulate_strategy.modulate()
        
    def __validate_input(self, matrix: PixelMatrix) -> None:
        """
        校验输入像素矩阵是否符合屏幕配置要求
        
        校验项:
        - 矩阵分辨率必须与屏幕配置一致
        
        Args:
            matrix: 待校验的像素矩阵
            
        Raises:
            ValueError: 模式不支持或分辨率不匹配
        """
        
        # 校验分辨率是否匹配屏幕
        if (matrix.width, matrix.height) != (self.__screen_config.width, self.__screen_config.height):
            raise ValueError(
                f"矩阵分辨率与屏幕不匹配:矩阵({matrix.width}x{matrix.height}),"
                f"屏幕({self.__screen_config.width}x{self.__screen_config.height})"
            )


