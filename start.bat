@echo off
chcp 65001 >nul
title vertex2api-golang

cd /d "%~dp0"

:: 检查 .env 文件
if not exist ".env" (
    if exist ".env.example" (
        echo 未找到 .env 文件，正在从 .env.example 复制...
        copy .env.example .env
        echo 请编辑 .env 文件配置你的 API Keys
        pause
        exit /b 1
    ) else (
        echo 错误: 未找到 .env 或 .env.example 文件
        pause
        exit /b 1
    )
)

:: 检查可执行文件
if not exist "vertex2api.exe" (
    echo 未找到可执行文件，正在编译...

    where go >nul 2>nul
    if %errorlevel% neq 0 (
        echo 错误: 未安装 Go，请先安装 Go 1.21+
        pause
        exit /b 1
    )

    go build -o vertex2api.exe ./cmd/server

    if %errorlevel% neq 0 (
        echo 编译失败
        pause
        exit /b 1
    )

    echo 编译成功
)

:: 启动服务
echo 正在启动 vertex2api-golang...
echo -----------------------------------

vertex2api.exe

pause
