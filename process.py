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


def download_model():
    if os.path.exists(MODEL_PATH):
        print("Model already exists.", flush=True)
        return

    print("Downloading BiRefNet model...", flush=True)

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_PATH,
    )

    print("Model download completed.", flush=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def run_birefnet(session, image):
    print("Preparing image...", flush=True)

    input_shape = session.get_inputs()[0].shape

    input_width = int(input_shape[3])
    input_height = int(input_shape[2])

    print(
        f"Model input: {input_width}x{input_height}",
        flush=True,
    )

    # Resize for model
    input_image = cv2.resize(
        image,
        (input_width, input_height),
    )

    # BGR -> RGB
    input_image = cv2.cvtColor(
        input_image,
        cv2.COLOR_BGR2RGB,
    )

    # Normalize
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

    # HWC -> CHW
    input_image = input_image.transpose(
        2,
        0,
        1,
    )

    # Add batch dimension
    input_image = np.expand_dims(
        input_image,
        axis=0,
    ).astype(np.float32)

    input_name = session.get_inputs()[0].name

    print("Running BiRefNet inference...", flush=True)

    start = time.time()

    result = session.run(
        None,
        {
            input_name: input_image
        },
    )

    elapsed = time.time() - start

    print(
        f"Inference completed: {elapsed:.1f} sec",
        flush=True,
    )

    # Official BiRefNet ONNX sample:
    # use the final output and apply sigmoid.
    mask = np.squeeze(result[-1])

    mask = sigmoid(mask)

    mask = (
        mask * 255
    ).astype(np.uint8)

    # Return mask to original image size
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
    print(
        "Creating white background...",
        flush=True,
    )

    # White image
    white = np.full_like(
        image,
        255,
        dtype=np.uint8,
    )

    # Convert mask to 0.0 - 1.0
    alpha = (
        mask.astype(np.float32)
        / 255.0
    )

    alpha = alpha[:, :, None]

    # Original product + white background
    result = (
        image.astype(np.float32) * alpha
        +
        white.astype(np.float32)
        * (1.0 - alpha)
    )

    result = np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)

    return result


def process_image(
    session,
    input_path,
    output_path,
):
    print("")
    print("=" * 60)
    print(
        f"Input: {input_path}",
        flush=True,
    )

    image = cv2.imread(
        input_path,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Cannot read image: {input_path}"
        )

    print(
        f"Original size: "
        f"{image.shape[1]}x{image.shape[0]}",
        flush=True,
    )

    mask = run_birefnet(
        session,
        image,
    )

    result = make_white_background(
        image,
        mask,
    )

    print(
        f"Saving: {output_path}",
        flush=True,
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
            f"Cannot save image: {output_path}"
        )

    print(
        "Completed successfully.",
        flush=True,
    )


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print(
        "=== BiRefNet Background Whitening ===",
        flush=True,
    )

    # Download model
    download_model()

    print(
        "Loading ONNX model...",
        flush=True,
    )

    # IMPORTANT:
    # GitHub Actions runner is CPU.
    # Do not initialize CUDA provider.
    session = onnxruntime.InferenceSession(
        MODEL_PATH,
        providers=[
            "CPUExecutionProvider"
        ],
    )

    print(
        "ONNX model loaded.",
        flush=True,
    )

    files = []

    for name in os.listdir(
        INPUT_DIR
    ):
        if name.lower().endswith(
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
            "No images found in input/"
        )

    print(
        f"Found {len(files)} image(s).",
        flush=True,
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
            session,
            input_path,
            output_path,
        )

    print("")
    print("=" * 60)
    print(
        "ALL PROCESSING COMPLETED",
        flush=True,
    )


if __name__ == "__main__":
    main()
