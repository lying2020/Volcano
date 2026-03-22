

# LML-diffusion-sampler
import os

# coco_data_path = "/home/liying/Documents/dataset/coco"
# coco_annotations_path = os.path.join(coco_data_path, "annotations")
# coco_val2014_images_path = os.path.join(coco_data_path, "val2014")

mme_data_path = "/home/liying/Documents/MLLMBenchMark/MME/extracted_data"

amber_data_path = "/home/liying/Documents/MLLMBenchMark/AMBER/data"

# llava_v15_7b_path = "/home/liying/Documents/llava-v1.5-7b"
# clip_vit_large_patch14_path = "/home/liying/Documents/clip-vit-large-patch14"
# anole_7b_v0_1_path = "/home/liying/Documents/Anole-7b-v0.1"
# volcano_7b_path = "/home/liying/Documents/volcano-7b"

coco_data_path = "/home1/cjl/MM_2026/dataset/ms_coco"
coco_annotations_path = os.path.join(coco_data_path, "annotations")
coco_val2014_images_path = os.path.join(coco_data_path, "val2014")

llava_v15_7b_path = "/home1/cjl/models/llava-v1.5-7b"
clip_vit_large_patch14_path = "/home1/cjl/models/clip-vit-large-patch14"
anole_7b_v0_1_path = "/home1/cjl/models/anole-7b"
volcano_7b_path = "/home1/cjl/models/volcano-7b"

# 推理测试图片目录（可选）：在此目录放 default.jpg 等，run_inference 会优先用作默认图
_project_dir = os.path.dirname(os.path.abspath(__file__))
test_images_dir = os.path.join(_project_dir, "test_images")