import os
import time
import urllib.request

import cv2
import numpy as np
import onnxruntime


MODEL_URL = (
    "https://github.com/"
    "Kazuhito00/BiRefNet-ONNX-Sample/"
    "releases/download/v0.0.1/"
    "birefnet_1024x1024.onnx"
)

MODEL_PATH = "birefnet_1024x1024.onnx"

INPUT_DIR = "input"
OUTPUT_DIR = "output"

CANVAS_SIZE = 1024
PRODUCT_MAX_SIZE = 924


def download_model():
    if os.path.exists(MODEL_PATH):
        print("Model already exists.")
        return

    print("Downloading BiRefNet ONNX model...")
    print(MODEL_URL)

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_PATH,
    )

    print("Model download completed.")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def create_mask(session, image):
    # モデルの入力サイズを取得
    input_shape = session.get_inputs()[0].shape

    input_width = int(input_shape[3])
    input_height = int(input_shape[2])

    # BiRefNet公式ONNXサンプルと同じ前処理
    input_image = cv2.resize(
        image,
        dsize=(input_width, input_height),
    )

    input_image = cv2.cvtColor(
        input_image,
        cv2.COLOR_BGR2RGB,
    )

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    input_image = (
        input_image.astype(np.float32) / 255.0
    )

    input_image = (
        input_image - mean
    ) / std

    input_image = input_image.transpose(
        2,
        0,
        1,
    )

    input_image = np.expand_dims(
        input_image,
        axis=0,
    )

    input_image = input_image.astype(
        np.float32
    )

    # 推論
    input_name = session.get_inputs()[0].name

    result = session.run(
        None,
        {
            input_name: input_image
        },
    )

    # 公式サンプルと同じく最後の出力をマスクとして使用
    mask = np.squeeze(result[-1])

    mask = sigmoid(mask)

    mask = (
        mask * 255
    ).astype(np.uint8)

    # 元画像サイズへ戻す
    mask = cv2.resize(
        mask,
        (
            image.shape[1],
            image.shape[0],
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    return mask


def make_white_background(image, mask):
    """
    商品をマスクで切り抜き、
    商品の外接矩形が924×924以内になるように
    1024×1024の白キャンバス中央へ配置する。
    """

    # マスクから商品領域を取得
    _, binary = cv2.threshold(
        mask,
        20,
        255,
        cv2.THRESH_BINARY,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise RuntimeError(
            "商品領域を検出できませんでした。"
        )

    # 最大輪郭を商品として扱う
    largest = max(
        contours,
        key=cv2.contourArea,
    )

    x, y, w, h = cv2.boundingRect(
        largest
    )

    if w <= 0 or h <= 0:
        raise RuntimeError(
            "商品領域のサイズが不正です。"
        )

    # 商品部分を切り出す
    crop_image = image[
        y:y + h,
        x:x + w,
    ]

    crop_mask = mask[
        y:y + h,
        x:x + w,
    ]

    # 商品を924×924以内に収める
    scale = min(
        PRODUCT_MAX_SIZE / w,
        PRODUCT_MAX_SIZE / h,
    )

    new_w = max(
        1,
        int(round(w * scale)),
    )

    new_h = max(
        1,
        int(round(h * scale)),
    )

    crop_image = cv2.resize(
        crop_image,
        (new_w, new_h),
        interpolation=cv2.INTER_LANCZOS4,
    )

    crop_mask = cv2.resize(
        crop_mask,
        (new_w, new_h),
        interpolation=cv2.INTER_LANCZOS4,
    )

    # 白キャンバス
    canvas = np.full(
        (
            CANVAS_SIZE,
            CANVAS_SIZE,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    # 中央配置
    offset_x = (
        CANVAS_SIZE - new_w
    ) // 2

    offset_y = (
        CANVAS_SIZE - new_h
    ) // 2

    roi = canvas[
        offset_y:offset_y + new_h,
        offset_x:offset_x + new_w,
    ]

    alpha = (
        crop_mask.astype(np.float32)
        / 255.0
    )

    alpha = alpha[:, :, None]

    composite = (
        crop_image.astype(np.float32)
        * alpha
        +
        roi.astype(np.float32)
        * (1.0 - alpha)
    )

    composite = np.clip(
        composite,
        0,
        255,
    ).astype(np.uint8)

    canvas[
        offset_y:offset_y + new_h,
        offset_x:offset_x + new_w,
    ] = composite

    return canvas


def process_image(
    input_path,
    output_path,
    session,
):
    print("")
    print("=" * 60)
    print("Processing:", input_path)

    image = cv2.imread(
        input_path,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"画像を読み込めません: {input_path}"
        )

    start = time.time()

    mask = create_mask(
        session,
        image,
    )

    result = make_white_background(
        image,
        mask,
    )

    success = cv2.imwrite(
        output_path,
        result,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )

    if not success:
        raise RuntimeError(
            f"画像を書き込めません: {output_path}"
        )

    elapsed = time.time() - start

    print(
        f"Completed: {output_path}"
    )

    print(
        f"Processing time: {elapsed:.1f} sec"
    )


def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    download_model()

    print("Loading BiRefNet...")

    session = onnxruntime.InferenceSession(
        MODEL_PATH,
        providers=[
            "CPUExecutionProvider"
        ],
    )

    print("BiRefNet loaded.")

    files = []

    for name in os.listdir(
        INPUT_DIR
    ):
        lower = name.lower()

        if lower.endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            )
        ):
            files.append(name)

    if not files:
        raise RuntimeError(
            "inputフォルダに画像がありません。"
        )

    print(
        f"Images found: {len(files)}"
    )

    for name in files:
        input_path = os.path.join(
            INPUT_DIR,
            name,
        )

        base = os.path.splitext(
            name
        )[0]

        output_path = os.path.join(
            OUTPUT_DIR,
            base + "_birefnet.jpg",
        )

        process_image(
            input_path,
            output_path,
            session,
        )

    print("")
    print("=" * 60)
    print("ALL DONE")


if __name__ == "__main__":
    main()
