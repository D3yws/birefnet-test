import os
import io
import cv2
import numpy as np
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import tempfile
from PIL import Image
from PIL.ExifTags import TAGS

ORIENTATION_TO_ROTATION = {
    3: cv2.ROTATE_180,
    6: cv2.ROTATE_90_CLOCKWISE,
    8: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def get_drive_service():
    creds = Credentials(
        token=None,
        refresh_token=os.environ["GOOGLE_REFRESH_TOKEN"],
        client_id=os.environ["GOOGLE_CLIENT_ID"],
        client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
        token_uri="https://oauth2.googleapis.com/token",
    )
    return build("drive", "v3", credentials=creds)


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
    res = service.files().list(q=query, fields="files(id, name)").execute()
    return res.get("files", [])


def download_image_bytes(service, file_id):
    request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    return buf.read()


def get_orientation(raw_bytes):
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        exif = img._getexif()
        if exif:
            for tag_id, value in exif.items():
                if TAGS.get(tag_id) == "Orientation":
                    return value
    except Exception:
        pass
    return 1


def overwrite_image(service, file_id, image):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    media = MediaFileUpload(tmp_path, mimetype="image/jpeg")
    service.files().update(fileId=file_id, media_body=media).execute()
    os.remove(tmp_path)


def rotate_image(raw_bytes, rotation):
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.rotate(image, rotation)


def main():
    print("=== Drive内画像 EXIF回転補正 ===", flush=True)

    service = get_drive_service()
    source_id = os.environ["DRIVE_WHITE_SOURCE_ID"]

    subfolders = list_subfolders(service, source_id)
    if not subfolders:
        raise RuntimeError("エラー：サブフォルダが見つかりませんでした。DRIVE_WHITE_SOURCE_IDを確認してください。")
    print(f"サブフォルダ数：{len(subfolders)}個", flush=True)

    total_fixed = 0
    total_skip = 0
    total_error = 0

    for folder in subfolders:
        print(f"\n=== フォルダ：{folder['name']} ===", flush=True)
        images = list_images(service, folder["id"])

        # ファイル一覧をname→{id}の辞書に変換
        file_map = {f["name"]: f["id"] for f in images}

        # 元画像（_white.jpgでないもの）を対象にEXIF確認
        originals = [f for f in images if "_white.jpg" not in f["name"]]
        print(f"元画像数：{len(originals)}件", flush=True)

        for f in originals:
            name = f["name"]
            base = os.path.splitext(name)[0]
            white_name = base + "_white.jpg"

            print(f"確認中：{name}", flush=True)

            raw = download_image_bytes(service, f["id"])
            orientation = get_orientation(raw)

            if orientation not in ORIENTATION_TO_ROTATION:
                print(f"  スキップ（補正不要 Orientation={orientation}）", flush=True)
                total_skip += 1
                continue

            rotation = ORIENTATION_TO_ROTATION[orientation]
            print(f"  Orientation={orientation} 回転補正します", flush=True)

            # 元画像を回転して上書き
            rotated = rotate_image(raw, rotation)
            if rotated is None:
                print(f"  エラー：元画像を読み込めませんでした：{name}", flush=True)
                total_error += 1
                continue
            overwrite_image(service, f["id"], rotated)
            print(f"  元画像補正完了：{name}", flush=True)
            total_fixed += 1

            # 対応する_white.jpgがあれば同様に回転して上書き
            if white_name in file_map:
                white_id = file_map[white_name]
                white_raw = download_image_bytes(service, white_id)
                white_rotated = rotate_image(white_raw, rotation)
                if white_rotated is None:
                    print(f"  エラー：_white.jpgを読み込めませんでした：{white_name}", flush=True)
                    total_error += 1
                else:
                    overwrite_image(service, white_id, white_rotated)
                    print(f"  _white.jpg補正完了：{white_name}", flush=True)
            else:
                print(f"  _white.jpgなし（スキップ）：{white_name}", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"補正完了：{total_fixed}件 / スキップ：{total_skip}件 / エラー：{total_error}件", flush=True)


if __name__ == "__main__":
    main()
