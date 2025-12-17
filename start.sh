#!/bin/bash

# vertex2api-golang 启动脚本 (Linux/Termux)

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 检查 .env 文件
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "未找到 .env 文件，正在从 .env.example 复制..."
        cp .env.example .env
        echo "请编辑 .env 文件配置你的 API Keys"
        exit 1
    else
        echo "错误: 未找到 .env 或 .env.example 文件"
        exit 1
    fi
fi

# 检测系统架构
ARCH=$(uname -m)
OS=$(uname -s)

# 设置可执行文件名
BINARY="vertex2api"

# 检查是否需要编译
if [ ! -f "$BINARY" ]; then
    echo "未找到可执行文件，正在编译..."

    # 检查 Go 是否安装
    if ! command -v go &> /dev/null; then
        echo "错误: 未安装 Go，请先安装 Go 1.21+"
        echo "Termux: pkg install golang"
        echo "Ubuntu/Debian: sudo apt install golang-go"
        exit 1
    fi

    go build -o "$BINARY" ./cmd/server

    if [ $? -ne 0 ]; then
        echo "编译失败"
        exit 1
    fi

    chmod +x "$BINARY"
    echo "编译成功"
fi

# 启动服务
echo "正在启动 vertex2api-golang..."
echo "系统: $OS ($ARCH)"
echo "-----------------------------------"

./"$BINARY"
