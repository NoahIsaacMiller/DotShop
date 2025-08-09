# 点阵小铺 (DotShop)


一款专注于屏幕取模的实用工具，轻松将文字、图像、动态GIF和视频转换为适配多种屏幕的点阵数据，支持多语言代码输出，让嵌入式屏幕显示开发更高效。

## 🌟 核心功能

- **多类型输入支持**：
  - 文字：字符串、TXT文件、TTF字体解析
  - 图像：PNG、JPG、BMP等静态格式
  - 动态内容：GIF动图、MP4/AVI等视频文件

- **多屏幕适配**：
  - 预设多种常见屏幕尺寸（128×64 OLED、240×240 LCD、16×16 LED矩阵等）
  - 支持自定义屏幕参数（分辨率、色彩模式、扫描方式）
  - 兼容单色、灰度、彩色多种显示模式

- **多语言输出**：
  - C语言数组
  - Python列表/字节流

- **便捷工具特性**：
  - 可视化预览取模效果
  - 批量处理多文件
  - 自定义取模参数（扫描方向、位序、对齐方式）
  - 插件扩展系统

## 🚀 快速开始

### 环境要求

- Python 3.9+
- 依赖库：见 `requirements.txt`

### 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/yourusername/dotshop.git
   cd dotshop
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 启动程序：
   ```bash
   # 图形界面模式
   python main.py
   
   # 命令行模式
   python main.py --cli
   ```

## 📖 使用指南

### 基础流程

1. 选择输入类型（文字/图像/GIF/视频）并加载内容
2. 在配置面板选择目标屏幕型号或自定义参数
3. 调整取模参数（色彩模式、扫描方向等）
4. 预览取模效果并确认
5. 选择输出语言，生成并复制代码

### 命令行示例

批量处理文件夹中的所有图像并生成C语言代码：
```
python main.py --cli \
  --input ./images \
  --output ./output \
  --screen oled_128x64 \
  --lang c \
  --batch
```
## 🛠️ 项目结构

遵循六边形架构设计，实现高内聚低耦合：
```
dotshop/
├── dotshop/core/          # 核心业务逻辑（取模算法、帧管理等）
├── dotshop/ports/         # 端口接口定义（输入、输出、配置等）
├── dotshop/adapters/      # 适配器实现（对接外部依赖）
├── dotshop/ui/            # 交互层（GUI和CLI）
├── dotshop/utils/         # 通用工具函数
├── configs/               # 屏幕配置和默认设置
├── plugins/               # 扩展插件
├── tests/                 # 单元测试和集成测试
└── docs/                  # 文档和资源
```
## 🔌 扩展开发

### 新增屏幕类型

1. 在 `configs/screens/` 目录下创建新的JSON配置文件
2. 配置参数示例：
   ```json
   {
     "id": "lcd_320x240",
     "name": "320×240 LCD屏幕",
     "resolution": {"width": 320, "height": 240},
     "color_mode": "rgb565",
     "scan_direction": "row",
     "bit_order": "msb_first",
     "byte_alignment": 16
   }
   ```

### 新增输出语言

1. 在 `dotshop/adapters/output/` 目录下创建新的适配器
2. 实现 `OutputPort` 接口的 `generate()` 方法

### 开发插件

1. 在 `plugins/` 目录下创建插件文件
2. 使用插件注册装饰器：
   ```python
   from dotshop.core.plugins import register_plugin

   @register_plugin("custom_compressor")
   def compress_data(data: bytes) -> bytes:
       # 实现自定义压缩逻辑
       return compressed_data
   ```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开 Pull Request

