# Deepseek2API2

## 运行
- 设置环境变量 `DEEPSEEK_AUTH_TOKEN`
- (可选) 设置 RAP 远程配置：
  - `RAP_URL`: 远程 RAP 文件 URL
  - `RAP_KEY`: 解密密钥
  - `RAP_UPDATE_TIME`: 自动拉取间隔（分钟），默认为 0（不自动拉取）
- 使用 uv 运行：
  - `uv run -m deepseek_client`
  - 或安装脚本后运行 `deepseek-client`

## Docker 运行
- 构建镜像：
  ```bash
  docker build -t deepseek2api2 .
  ```
- 运行容器：
  ```bash
  docker run -p 8000:8000 -e DEEPSEEK_AUTH_TOKEN=your_token deepseek2api2
  ```

## 目录结构
- 核心包：`src/deepseek_client/` 包含 CLI、HTTP、PoW、配置与常量
- 资产：`src/deepseek_client/assets/`（开发阶段可使用根目录 WASM 回退）
- 测试：`tests/` 最小化冒烟与 PoW 健全性测试
- 示例：`examples/basic_chat.py`
- 逆向产物：`artifacts/`（非运行必须）

## 说明
- `x-hif-leim` 为 HAR 中值，生成逻辑未知，暂以常量保留
- WASM 文件在打包时应置于 `assets/`，开发阶段代码将回退到项目根目录查找
