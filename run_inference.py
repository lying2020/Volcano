#!/usr/bin/env python3
"""
Volcano 单图单问推理脚本。
默认使用 project.py 中的 volcano_7b_path、llava_v15_7b_path 以及 COCO val2014 示例图。
用法示例：
  python run_inference.py
  python run_inference.py --image_path /path/to/image.jpg --question "图片中有什么？"
  python run_inference.py --model_path kaist-ai/volcano-7b --model_base liuhaotian/llava-v1.5-7b \\
    --image_path /path/to/image.jpg --question "图片中有什么？"
"""
import argparse
import os
import torch
from PIL import Image

import project
_DEFAULT_MODEL_PATH = getattr(project, "volcano_7b_path", None)
_DEFAULT_MODEL_BASE = getattr(project, "llava_v15_7b_path", None)
_test_dir = getattr(project, "test_images_dir", None)
_coco_path = getattr(project, "coco_val2014_images_path", None)
_default_img = None
if _test_dir and os.path.isfile(os.path.join(_test_dir, "default.jpg")):
    _default_img = os.path.join(_test_dir, "default.jpg")
elif _test_dir and os.path.isfile(os.path.join(_test_dir, "default.png")):
    _default_img = os.path.join(_test_dir, "default.png")
if _default_img is None and _coco_path:
    _default_img = os.path.join(_coco_path, "COCO_val2014_000000000042.jpg")
    _default_img = _default_img if os.path.isfile(_default_img) else None
_DEFAULT_IMAGE_PATH = _default_img

_DEFAULT_QUESTION = "What is in this image?"

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)


def load_image(image_path: str) -> Image.Image:
    if image_path.startswith("http://") or image_path.startswith("https://"):
        import requests
        from io import BytesIO
        resp = requests.get(image_path)
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(image_path).convert("RGB")


def extract_image_features(image, model, image_processor):
    images = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().cuda()
    if isinstance(images, list) or (hasattr(images, "ndim") and images.ndim == 5):
        concat_images = torch.cat([im for im in images], dim=0)
        image_features = model.encode_images(concat_images)
        split_sizes = [im.shape[0] for im in images]
        image_features = torch.split(image_features, split_sizes, dim=0)
        image_features = [x.flatten(0, 1) for x in image_features]
    else:
        image_features = model.encode_images(images)
    return image_features


def generate_one(
    question_text: str,
    image_features,
    tokenizer,
    model,
    model_name: str,
    image_processor,
    context_len: int,
    conv_mode: str,
    temperature: float = 0.2,
    max_new_tokens: int = 1024,
) -> str:
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question_text
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question_text

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            image_features=image_features,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    return outputs.strip()


def main():
    parser = argparse.ArgumentParser(description="Volcano 单图单问推理")
    parser.add_argument(
        "--model_path",
        type=str,
        default=_DEFAULT_MODEL_PATH,
        help="Volcano 权重路径或 HuggingFace 仓库名（默认从 project.volcano_7b_path 读取）",
    )
    parser.add_argument(
        "--model_base",
        type=str,
        default=_DEFAULT_MODEL_BASE,
        help="LLaVA-1.5 基座路径或 HuggingFace 仓库名（默认从 project.llava_v15_7b_path 读取）",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=_DEFAULT_IMAGE_PATH,
        help="输入图片路径或 URL（默认 COCO val2014 示例图，需 project.py 中 coco_val2014_images_path 可用）",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=_DEFAULT_QUESTION,
        help="问题文本（默认: What is in this image?）",
    )
    parser.add_argument(
        "--max_revision_rounds",
        type=int,
        default=3,
        help="自反馈修订最大轮数（默认 3）",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--conv_mode", type=str, default=None)
    args = parser.parse_args()

    if not args.model_path or not args.model_base:
        parser.error("未配置模型路径。请在 project.py 中设置 volcano_7b_path 与 llava_v15_7b_path，或通过 --model_path / --model_base 传入。")
    if not args.image_path or not os.path.isfile(args.image_path):
        parser.error(
            "未指定有效图片。请通过 --image_path 传入本地图片路径；"
            "或将测试图放在 test_images/ 下并命名为 default.jpg / default.png，或设置 project.py 中 coco_val2014_images_path。"
        )

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    conv_mode = args.conv_mode or "llava_v1"

    image = load_image(args.image_path)
    image_features = extract_image_features(image, model, image_processor)

    # 初始回答
    initial = generate_one(
        args.question,
        image_features,
        tokenizer,
        model,
        model_name,
        image_processor,
        context_len,
        conv_mode,
        temperature=args.temperature,
    )
    print("[Initial answer]", initial)
    output = initial

    for r in range(args.max_revision_rounds):
        # Critique
        fq = (
            "Generate the feedback given initial answer referring to question and image.\n"
            f"Question: {args.question}\nInitial answer: {output}"
        )
        feedback = generate_one(
            fq,
            image_features,
            tokenizer,
            model,
            model_name,
            image_processor,
            context_len,
            conv_mode,
            temperature=args.temperature,
        )
        print(f"[Round {r+1} feedback]", feedback)

        # Revise
        rq = (
            "Adjust the initial response considering the feedback and image.\n"
            f"Question: {args.question}\nInitial answer: {output}\nFeedback: {feedback}"
        )
        revision = generate_one(
            rq,
            image_features,
            tokenizer,
            model,
            model_name,
            image_processor,
            context_len,
            conv_mode,
            temperature=args.temperature,
        )
        print(f"[Round {r+1} revision]", revision)

        # Decide
        decide_prompt = (
            f"{args.question}\nA. {output}\nB. {revision}\n"
            "Answer with the option's letter from the given choices directly."
        )
        decision = generate_one(
            decide_prompt,
            image_features,
            tokenizer,
            model,
            model_name,
            image_processor,
            context_len,
            conv_mode,
            temperature=args.temperature,
            max_new_tokens=16,
        )
        print(f"[Round {r+1} decision]", decision)
        if "b" in decision.lower():
            output = revision
        elif "a" in decision.lower():
            break

    print("\n[Final answer]", output)


if __name__ == "__main__":
    main()
