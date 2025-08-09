#!/bin/bash

# 点阵小铺(DotShop)项目结构创建脚本
# 功能：自动生成项目所需的所有目录和初始文件

# 定义项目根目录名称
PROJECT_ROOT="dotshop"

# 打印欢迎信息
echo "============================================="
echo "开始创建 点阵小铺(DotShop) 项目结构..."
echo "项目根目录: $PROJECT_ROOT"
echo "============================================="

# 创建项目根目录
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT" || { echo "无法进入项目目录，创建失败"; exit 1; }

# 创建核心包目录及子模块
mkdir -p dotshop/{core,ports,adapters/{input,output,config,preview},ui/{gui/{resources},},utils}

# 创建核心层文件
touch dotshop/__init__.py
touch dotshop/core/__init__.py
touch dotshop/core/modulator.py
touch dotshop/core/frame_manager.py
touch dotshop/core/format_converter.py
touch dotshop/core/config_validator.py

# 创建端口层文件
touch dotshop/ports/__init__.py
touch dotshop/ports/input_port.py
touch dotshop/ports/output_port.py
touch dotshop/ports/config_port.py
touch dotshop/ports/preview_port.py

# 创建输入适配器文件
touch dotshop/adapters/__init__.py
touch dotshop/adapters/input/__init__.py
touch dotshop/adapters/input/text_adapter.py
touch dotshop/adapters/input/image_adapter.py
touch dotshop/adapters/input/gif_adapter.py
touch dotshop/adapters/input/video_adapter.py

# 创建输出适配器文件
touch dotshop/adapters/output/__init__.py
touch dotshop/adapters/output/c_adapter.py
touch dotshop/adapters/output/python_adapter.py
touch dotshop/adapters/output/js_adapter.py

# 创建配置适配器文件
touch dotshop/adapters/config/__init__.py
touch dotshop/adapters/config/screen_config.py
touch dotshop/adapters/config/user_config.py

# 创建预览适配器文件
touch dotshop/adapters/preview/__init__.py
touch dotshop/adapters/preview/static_preview.py
touch dotshop/adapters/preview/dynamic_preview.py

# 创建UI层文件
touch dotshop/ui/__init__.py
touch dotshop/ui/cli.py
touch dotshop/ui/gui/__init__.py
touch dotshop/ui/gui/main_window.py
touch dotshop/ui/gui/widgets.py

# 创建工具函数层文件
touch dotshop/utils/__init__.py
touch dotshop/utils/image_utils.py
touch dotshop/utils/file_utils.py
touch dotshop/utils/log_utils.py

# 创建配置文件目录及示例配置
mkdir -p configs/screens
touch configs/default_settings.json
touch configs/screens/oled_128x64.json
touch configs/screens/lcd_240x240.json
touch configs/screens/led_matrix_16x16.json

# 创建插件目录
mkdir -p plugins
touch plugins/__init__.py
touch plugins/example_plugin.py

# 创建测试目录
mkdir -p tests/{test_core,test_adapters,test_ui}
touch tests/__init__.py

# 创建文档目录
mkdir -p docs/api_docs
touch docs/user_guide.md
touch docs/dev_guide.md

# 创建项目入口和配置文件
touch main.py
touch requirements.txt
touch setup.py

# 打印完成信息
echo "============================================="
echo "项目结构创建完成！"
echo "位置: $(pwd)"
echo "目录结构已按照六边形架构设计，包含："
echo "- 核心业务逻辑层(core)"
echo "- 端口接口层(ports)"
echo "- 适配器层(adapters)"
echo "- 交互层(ui)"
echo "- 配置文件、插件、测试和文档目录"
echo "============================================="
