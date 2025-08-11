from typing import Iterable, Tuple, List, Type, Dict, ABC, abstractmethod
from data_structures import ColorMode, ScanDirection, BitOrder


class PixelScanOrder:
    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def get_coordinate_sequence(self) -> Iterable[Tuple[int, int]]:
        raise NotImplementedError

    def is_page_based(self) -> bool:
        return False


class RowScanOrder(PixelScanOrder):
    def get_coordinate_sequence(self) -> Iterable[Tuple[int, int]]:
        for y in range(self.height):
            for x in range(self.width):
                yield (x, y)


class ReverseRowScanOrder(PixelScanOrder):
    def get_coordinate_sequence(self) -> Iterable[Tuple[int, int]]:
        for y in range(self.height):
            for x in range(self.width - 1, -1, -1):
                yield (x, y)


class ColumnScanOrder(PixelScanOrder):
    def get_coordinate_sequence(self) -> Iterable[Tuple[int, int]]:
        for x in range(self.width):
            for y in range(self.height):
                yield (x, y)


class ReverseColumnScanOrder(PixelScanOrder):
    def get_coordinate_sequence(self) -> Iterable[Tuple[int, int]]:
        for x in range(self.width):
            for y in range(self.height - 1, -1, -1):
                yield (x, y)


class PageBasedScanOrder(PixelScanOrder):
    def __init__(self, width: int, height: int) -> None:
        if height % 8 != 0:
            raise ValueError(f"页模式高度必须是8的倍数，实际为{height}")
        super().__init__(width, height)

    @property
    def page_count(self) -> int:
        return self.height // 8

    def is_page_based(self) -> bool:
        return True

    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, value: int) -> None:
        if value % 8 != 0:
            raise ValueError(f"页模式高度必须是8的倍数，实际为{value}")
        self._height = value


class PageColumnScanOrder(PageBasedScanOrder):
    def get_coordinate_sequence(self) -> Iterable[Tuple[int, int]]:
        for page in range(self.page_count):
            for x in range(self.width):
                for y_offset in range(8):
                    yield (x, page * 8 + y_offset)


class ReversePageColumnScanOrder(PageBasedScanOrder):
    def get_coordinate_sequence(self) -> Iterable[Tuple[int, int]]:
        for page in range(self.page_count):
            for x in range(self.width):
                for y_offset in range(7, -1, -1):
                    yield (x, page * 8 + y_offset)


class PixelBitCoder(ABC):
    @abstractmethod
    def encode_8pixels(self, pixels: List[int]) -> int:
        raise NotImplementedError

    def pad_to_8(self, pixels: List[int]) -> List[int]:
        if len(pixels) > 8:
            raise ValueError(f"像素列表长度不能超过8，实际为{len(pixels)}")
        return pixels + [0] * (8 - len(pixels))

    def decode_byte(self, byte: int) -> List[int]:
        bits = []
        for i in range(8):
            bits.append((byte & (0x80 >> i)) >> (7 - i))
        return bits

    def reencode_byte(self, byte: int) -> int:
        return self.encode_8pixels(self.decode_byte(byte))


class MSBFirstCoder(PixelBitCoder):
    def encode_8pixels(self, pixels: List[int]) -> int:
        if len(pixels) != 8:
            raise ValueError(f"需要8个像素，实际为{len(pixels)}")
        
        for pixel in pixels:
            if pixel not in (0, 1):
                raise ValueError(f"二值模式像素值必须为0或1，实际为{pixel}")
        
        result = 0
        for i in range(8):
            result |= pixels[i] << (7 - i)
        return result


class LSBFirstCoder(PixelBitCoder):
    def encode_8pixels(self, pixels: List[int]) -> int:
        if len(pixels) != 8:
            raise ValueError(f"需要8个像素，实际为{len(pixels)}")
        
        for pixel in pixels:
            if pixel not in (0, 1):
                raise ValueError(f"二值模式像素值必须为0或1，实际为{pixel}")
        
        result = 0
        for i in range(8):
            result |= pixels[i] << i
        return result


class PixelModulator(ABC):
    def __init__(self, 
                matrix, 
                scan_order_cls: Type[PixelScanOrder],
                bit_coder_cls: Type[PixelBitCoder]):
        self._matrix = matrix
        self._scan_order = scan_order_cls(matrix.width, matrix.height)
        self._bit_coder = bit_coder_cls()

    @abstractmethod
    def modulate(self) -> bytes:
        raise NotImplementedError


class MonoPixelModulator(PixelModulator):
    def modulate(self) -> bytes:
        byte_data = []
        if self._scan_order.is_page_based():
            page_scan_order: PageBasedScanOrder = self._scan_order
            for page in range(page_scan_order.page_count):
                for x in range(page_scan_order.width):
                    pixels = [
                        self._matrix.get_pixel(x, page*8 + y_offset)[0]
                        for y_offset in range(8)
                    ]
                    byte_data.append(self._bit_coder.encode_8pixels(pixels))
        else:
            pixel_values = [
                self._matrix.get_pixel(x, y)[0]
                for x, y in self._scan_order.get_coordinate_sequence()
            ]
            for i in range(0, len(pixel_values), 8):
                group = self._bit_coder.pad_to_8(pixel_values[i:i+8])
                byte_data.append(self._bit_coder.encode_8pixels(group))
        return bytes(byte_data)


class GrayscaleModulator(PixelModulator):
    def modulate(self) -> bytes:
        byte_data = []
        for x, y in self._scan_order.get_coordinate_sequence():
            val = self._matrix.get_pixel(x, y)[0]
            byte_data.append(val & 0xFF)
        return bytes(byte_data)


class RgbModulator(PixelModulator):
    pass


class Rgb565Modulator(RgbModulator):
    def modulate(self) -> bytes:
        byte_data = []
        for x, y in self._scan_order.get_coordinate_sequence():
            value = self._matrix.get_pixel(x, y)[0]
            byte_data.append((value >> 8) & 0xFF)
            byte_data.append(value & 0xFF)
        return bytes(byte_data)


class Rgb888Modulator(RgbModulator):
    def modulate(self) -> bytes:
        byte_data = []
        if self._scan_order.is_page_based():
            page_scan_order: PageBasedScanOrder = self._scan_order
            for page in range(page_scan_order.page_count):
                for x in range(page_scan_order.width):
                    for y_offset in range(8):
                        y = page * 8 + y_offset
                        r, g, b = self._matrix.get_pixel(x, y)
                        if isinstance(self._bit_coder, LSBFirstCoder):
                            byte_data.extend([
                                self._bit_coder.reencode_byte(r),
                                self._bit_coder.reencode_byte(g),
                                self._bit_coder.reencode_byte(b)
                            ])
                        else:
                            byte_data.extend([r, g, b])
        else:
            for x, y in self._scan_order.get_coordinate_sequence():
                r, g, b = self._matrix.get_pixel(x, y)
                if isinstance(self._bit_coder, LSBFirstCoder):
                    byte_data.extend([
                        self._bit_coder.reencode_byte(r),
                        self._bit_coder.reencode_byte(g),
                        self._bit_coder.reencode_byte(b)
                    ])
                else:
                    byte_data.extend([r, g, b])
        return bytes(byte_data)


class Rgb8888Modulator(RgbModulator):
    def modulate(self) -> bytes:
        byte_data = []
        if self._scan_order.is_page_based():
            page_scan_order: PageBasedScanOrder = self._scan_order
            for page in range(page_scan_order.page_count):
                for x in range(page_scan_order.width):
                    for y_offset in range(8):
                        y = page * 8 + y_offset
                        r, g, b, a = self._matrix.get_pixel(x, y)
                        if isinstance(self._bit_coder, LSBFirstCoder):
                            byte_data.extend([
                                self._bit_coder.reencode_byte(r),
                                self._bit_coder.reencode_byte(g),
                                self._bit_coder.reencode_byte(b),
                                self._bit_coder.reencode_byte(a)
                            ])
                        else:
                            byte_data.extend([r, g, b, a])
        else:
            for x, y in self._scan_order.get_coordinate_sequence():
                r, g, b, a = self._matrix.get_pixel(x, y)
                if isinstance(self._bit_coder, LSBFirstCoder):
                    byte_data.extend([
                        self._bit_coder.reencode_byte(r),
                        self._bit_coder.reencode_byte(g),
                        self._bit_coder.reencode_byte(b),
                        self._bit_coder.reencode_byte(a)
                    ])
                else:
                    byte_data.extend([r, g, b, a])
        return bytes(byte_data)


SCAN_ORDER_MAP: Dict[ScanDirection, Type[PixelScanOrder]] = {
    ScanDirection.HORIZONTAL: RowScanOrder,
    ScanDirection.HORIZONTAL_REVERSED: ReverseRowScanOrder,
    ScanDirection.VERTICAL: ColumnScanOrder,
    ScanDirection.VERTICAL_REVERSED: ReverseColumnScanOrder,
    ScanDirection.BY_PAGE: PageColumnScanOrder,
    ScanDirection.BY_PAGE_REVERSED: ReversePageColumnScanOrder
}

MODULATOR_MAP: Dict[ColorMode, Type[PixelModulator]] = {
    ColorMode.MONO: MonoPixelModulator,
    ColorMode.GRAY_8: GrayscaleModulator,
    ColorMode.RGB565: Rgb565Modulator,
    ColorMode.RGB_888: Rgb888Modulator,
    ColorMode.RGB_8888: Rgb8888Modulator
}


class DisplayDataGenerator:
    def __init__(self, config):
        self._config = config

    def generate(self, matrix) -> bytes:
        self._validate_input(matrix)
        
        scan_order_cls = SCAN_ORDER_MAP[self._config.scan_direction]
        bit_coder_cls = MSBFirstCoder if self._config.bit_order == BitOrder.MSB_FIRST else LSBFirstCoder
        
        modulator = MODULATOR_MAP[matrix.mode](
            matrix=matrix,
            scan_order_cls=scan_order_cls,
            bit_coder_cls=bit_coder_cls
        )
        return modulator.modulate()

    def _validate_input(self, matrix) -> None:
        if (matrix.width, matrix.height) != (self._config.width, self._config.height):
            raise ValueError(
                f"分辨率不匹配: 矩阵({matrix.width}x{matrix.height}) vs 屏幕({self._config.width}x{self._config.height})"
            )
        
        if matrix.mode not in self._config.supported_modes:
            raise ValueError(
                f"不支持的模式: {matrix.mode}, 支持模式: {self._config.supported_modes}"
            )
        
        scan_order_cls = SCAN_ORDER_MAP[self._config.scan_direction]
        if issubclass(scan_order_cls, PageBasedScanOrder) and matrix.height % 8 != 0:
            raise ValueError(f"页模式要求高度为8的倍数，实际为{matrix.height}")
