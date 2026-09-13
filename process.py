import os
import cv2
import numpy as np
from PIL import Image
import onnxruntime


MODEL_URL = (
    "https://huggingface.co/onnx-community/BiRefNet-ONNX/"
    "resolve/main/onnx/model_fp16.onnx"
)

MODEL_PATH = "model.onnx"
INPUT_DIR = "input"
OUTPUT_DIR = "output"


def download_model():
    if os.path.exists(MODEL_PATH):
        return

    import urllib.request

    print("Downloading BiRefNet ONNX model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def process_image(input_path, output_path, session):
    image = cv2.imread(input_path)

    if image is None:
        raise RuntimeError(f"画像を読み込めません: {input_path}")

    original_h, original_w = image.shape[:2]

    input_shape = session.get_inputs()[0].shape

    input_h = input_shape[2]
    input_w = input_shape[3]

    resized = cv2.resize(
        image,
        (input_w, input_h),
        interpolation=cv2.INTER_LINEAR,
    )

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    rgb = (rgb - mean) / std

    tensor = rgb.transpose(2, 0, 1)
    tensor = np.expand_dims(tensor, axis=0)
    tensor = tensor.astype(np.float32)

    input_name = session.get_inputs()[0].name

    result = session.run(
        None,
        {input_name: tensor},
    )

    mask = np.squeeze(result[-1])
    mask = sigmoid(mask)

    mask = (mask * 255).astype(np.uint8)

    mask = cv2.resize(
        mask,
        (original_w, original_h),
        interpolation=cv2.INTER_LINEAR,
    )

    # 白背景
    white = np.full_like(image, 255)

    alpha = mask.astype(np.float32) / 255.0
    alpha = alpha[:, :, None]

    result_image = (
        image.astype(np.float32) * alpha
        + white.astype(np.float32) * (1.0 - alpha)
    )

    result_image = np.clip(
        result_image,
        0,
        255,
    ).astype(np.uint8)

    cv2.imwrite(output_path, result_image)

    print(f"完成: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    download_model()

    session = onnxruntime.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )

    files = []

    for name in os.listdir(INPUT_DIR):
        lower = name.lower()

        if lower.endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            files.append(name)

    if not files:
        raise RuntimeError(
            "inputフォルダに画像がありません。"
        )

    for name in files:
        input_path = os.path.join(
            INPUT_DIR,
            name,
        )

        base = os.path.splitext(name)[0]

        output_path = os.path.join(
            OUTPUT_DIR,
            base + "_birefnet.jpg",
        )

        process_image(
            input_path,
            output_path,
            session,
        )


if __name__ == "__main__":
    main()
