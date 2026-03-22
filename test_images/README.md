# Volcano 推理测试图片目录

将用于测试的图片放在此目录下，使用 `run_inference.py` 时通过 `--image_path test_images/你的图片文件名` 指定。

## 支持的图片格式

脚本使用 PIL/Pillow 读取并转为 RGB，以下格式均可使用：

| 格式     | 常见扩展名     | 说明     |
|----------|----------------|----------|
| JPEG     | `.jpg`, `.jpeg`| 最常用   |
| PNG      | `.png`         | 支持透明通道（会转为 RGB） |
| WebP     | `.webp`        | 现代格式 |
| BMP      | `.bmp`         | 位图     |
| TIFF     | `.tiff`, `.tif`| 高分辨率 |
| GIF      | `.gif`         | 仅使用第一帧 |
| ICO      | `.ico`         | 图标     |
| PPM/PGM  | `.ppm`, `.pgm` | 少见     |

**建议**：优先使用 `.jpg` 或 `.png`，兼容性最好。

## 默认测试图

在本目录下放入 **default.jpg** 或 **default.png** 后，直接运行 `python run_inference.py` 会优先使用该图，无需每次指定 `--image_path`。

## 使用示例

```bash
# 使用本目录下的图片（任意支持的格式）
python run_inference.py --image_path test_images/sample.jpg --question "图片中有什么？"
python run_inference.py --image_path test_images/photo.png --question "Describe this image."

# 若已放置 default.jpg 或 default.png，可省略 --image_path
python run_inference.py --question "What is in this image?"
```

## 注意事项

- 图片会经 CLIP 预处理（如 336×336 等），过大或过小都会自动处理，无需事先裁剪。
- URL 也支持：`--image_path "https://example.com/image.jpg"`。
