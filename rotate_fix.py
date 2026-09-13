import os
import io
import cv2
import numpy as np
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import tempfile

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
        from PIL import Image
        from PIL.ExifTags import TAGS
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

    for folder in subfolders:
        print(f"\n=== フォルダ：{folder['name']} ===", flush=True)
        images = list_images(service, folder["id"])
        print(f"対象ファイル数：{len(images)}件", flush=True)

        for f in images:
            name = f["name"]
            print(f"確認中：{name}", flush=True)

            raw = download_image_bytes(service, f["id"])
            orientation = get_orientation(raw)

            if orientation not in ORIENTATION_TO_ROTATION:
                print(f"  スキップ（補正不要 Orientation={orientation}）", flush=True)
                total_skip += 1
                continue

            rotation = ORIENTATION_TO_ROTATION[orientation]
            arr = np.frombuffer(raw, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is None:
                print(f"  エラー：画像を読み込めませんでした：{name}", flush=True)
                continue

            rotated = cv2.rotate(image, rotation)
            overwrite_image(service, f["id"], rotated)
            print(f"  補正完了（Orientation={orientation}）：{name}", flush=True)
            total_fixed += 1

    print(f"\n{'=' * 60}", flush=True)
    print(f"補正完了：{total_fixed}件 / スキップ：{total_skip}件", flush=True)


if __name__ == "__main__":
    main()
