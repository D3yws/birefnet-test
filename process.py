import os
import time
import urllib.request
import cv2
import numpy as np
import onnxruntime
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import io
import tempfile

MODEL_URL = (
    "https://github.com/"
    "Kazuhito00/BiRefNet-ONNX-Sample/"
    "releases/download/v0.0.1/"
    "birefnet_1024x1024.onnx"
)
MODEL_PATH = "birefnet_1024x1024.onnx"


def get_drive_service():
    creds = Credentials(
        token=None,
        refresh_token=os.environ["GOOGLE_REFRESH_TOKEN"],
        client_id=os.environ["GOOGLE_CLIENT_ID"],
        client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
        token_uri="https://oauth2.googleapis.com/token",
    )
    return build("drive", "v3", credentials=creds)


def download_model():
    if os.path.exists(MODEL_PATH):
        print("Model already exists.", flush=True)
        return
    print("Downloading BiRefNet model...", flush=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model download completed.", flush=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def run_birefnet(session, image):
    print("Preparing image...", flush=True)
    input_shape = session.get_inputs()[0].shape
    input_width = int(input_shape[3])
    input_height = int(input_shape[2])
    print(f"Model input: {input_width}x{input_height}", flush=True)

    input_image = cv2.resize(image, (input_width, input_height))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    input_image = (input_image.astype(np.float32) / 255.0 - mean) / std
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32)

    input_name = session.get_inputs()[0].name
    print("Running BiRefNet inference...", flush=True)
    start = time.time()
    result = session.run(None, {input_name: input_image})
    elapsed = time.time() - start
    print(f"Inference completed: {elapsed:.1f} sec", flush=True)

    mask = np.squeeze(result[-1])
    mask = sigmoid(mask)
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(
        mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return mask


def make_white_background(image, mask):
    print("Creating white background...", flush=True)
    white = np.full_like(image, 255, dtype=np.uint8)
    alpha = mask.astype(np.float32) / 255.0
    alpha = alpha[:, :, None]
    result = (
        image.astype(np.float32) * alpha
        + white.astype(np.float32) * (1.0 - alpha)
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def list_subfolders(service, parent_id):
    query = (
        f"'{parent_id}' in parents"
        " and mimeType='application/vnd.google-apps.folder'"
        " and trashed=false"
    )
    res = service.files().list(q=query, fields="files(id, name)").execute()
    return res.get("files", [])


def list_images(service, folder_id):
    query = (
        f"'{folder_id}' in parents"
        " and mimeType contains 'image/'"
        " and trashed=false"
    )
    res = service.files().list(
        q=query, fields="files(id, name)"
    ).execute()
    # _white.jpg 済みはスキップ
    files = [
        f for f in res.get("files", [])
        if not f["name"].endswith("_white.jpg")
    ]
    return files


def download_image(service, file_id):
    request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    arr = np.frombuffer(buf.read(), dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image


def upload_image(service, folder_id, filename, image):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    media = MediaFileUpload(tmp_path, mimetype="image/jpeg")
    service.files().create(
        body={"name": filename, "parents": [folder_id]},
        media_body=media,
    ).execute()
    os.remove(tmp_path)


def main():
    print("=== BiRefNet Background Whitening ===", flush=True)

    download_model()

    print("Loading ONNX model...", flush=True)
    session = onnxruntime.InferenceSession(
        MODEL_PATH, providers=["CPUExecutionProvider"]
    )
    print("ONNX model loaded.", flush=True)

    service = get_drive_service()
    source_id = os.environ["DRIVE_WHITE_SOURCE_ID"]

    subfolders = list_subfolders(service, source_id)
    if not subfolders:
        raise RuntimeError("エラー：サブフォルダが見つかりませんでした。DRIVE_WHITE_SOURCE_IDを確認してください。")
    print(f"サブフォルダ数：{len(subfolders)}個", flush=True)

    total = 0
    for folder in subfolders:
        print(f"\n=== フォルダ：{folder['name']} ===", flush=True)
        images = list_images(service, folder["id"])
        print(f"対象ファイル数：{len(images)}件", flush=True)

        for f in images:
            name = f["name"]
            base = os.path.splitext(name)[0]
            out_name = base + "_white.jpg"

            print(f"\n処理中：{name}", flush=True)
            image = download_image(service, f["id"])
            if image is None:
                print(f"エラー：画像を読み込めませんでした：{name}", flush=True)
                continue

            mask = run_birefnet(session, image)
            result = make_white_background(image, mask)
            upload_image(service, folder["id"], out_name, result)
            print(f"保存完了：{out_name}", flush=True)
            total += 1

    print(f"\n{'=' * 60}", flush=True)
    print(f"全処理完了：{total}件", flush=True)


if __name__ == "__main__":
    main()
