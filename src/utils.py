import os
import io
from contextlib import contextmanager
from PIL import Image, ImageDraw, ImageFont, ImageOps
import requests
from io import BytesIO
import re
import transformers
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hw_decoders_any;none")

import cv2
import logging
import shutil
import subprocess

@contextmanager
def _suppress_stderr():
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)

@contextmanager
def _suppress_cv2_logging():
    if not hasattr(cv2, "getLogLevel") or not hasattr(cv2, "setLogLevel"):
        yield
        return

    previous_level = cv2.getLogLevel()
    try:
        cv2.setLogLevel(0)
        yield
    finally:
        cv2.setLogLevel(previous_level)

def _open_video_capture(video_path):
    with _suppress_stderr(), _suppress_cv2_logging():
        try:
            return cv2.VideoCapture(
                video_path,
                cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE],
            )
        except TypeError:
            return cv2.VideoCapture(video_path)

def create_file_list(root_dir):
    dir_contents = os.listdir(root_dir)
    path_list = [os.path.join(root_dir, file_path) for file_path in dir_contents]

    return path_list

def read_image(path):
    img = Image.open(path)

    img.show()

    print(f"Image size: {img.size}")

def process_image(path):
    img = Image.open(path)

    width, height = img.size
    if width < height:
        img = img.rotate(angle=90, resample=Image.BICUBIC, expand=1)
    if width > 2000:
        img = img.resize((1440, 1080))
    img.show()

    print(f"Image size: {img.size}")

    return img

def pad_vit_input(img, bbox):
    """
    Crops image to bbox, adds padding to make it square, and resizes.
    bbox format: [x_min, y_min, x_max, y_max]
    """
    if bbox == [-1, -1, -1, -1]:
        return img

    width, height = img.size
    
    # 1. Unpack and clamp bbox to image boundaries
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(width, int(x_max))
    y_max = min(height, int(y_max))
    
    # 2. Crop
    cropped_img = img.crop((x_min, y_min, x_max, y_max))
    
    # 3. Calculate Padding to change it to original size
    padding_l = max(0, x_min)
    padding_t = max(0, y_min)
    padding_r = max(0, width - x_max)
    padding_b = max(0, height - y_max)
    
    # 4. Apply Padding (Fill with 0, 128, or mean color)
    # Define padding: (left, top, right, bottom)
    padding = (padding_l, padding_t, padding_r, padding_b) 
    padded_img = ImageOps.expand(cropped_img, padding, fill='black')
    
    return padded_img

def concat_images(img1, img2, direction='horizontal', padding=0, bg_color=(0,0,0), save_path=None):
    """
    Concatenate two images side-by-side (horizontal) or top-bottom (vertical).

    Args:
        img1: file path or PIL.Image.Image for the first image.
        img2: file path or PIL.Image.Image for the second image.
        direction: 'horizontal' (default) or 'vertical'.
        padding: pixels between images (default 0).
        bg_color: background color tuple for padding areas (default black).
        save_path: if provided, save concatenated image to this path.

    Returns:
        PIL.Image.Image of the concatenated image.
    """
    # Load images if paths are provided
    im1 = Image.open(img1).convert('RGB') if isinstance(img1, str) else img1.convert('RGB')
    im2 = Image.open(img2).convert('RGB') if isinstance(img2, str) else img2.convert('RGB')

    w1, h1 = im1.size
    w2, h2 = im2.size

    if direction == 'horizontal':
        new_h = max(h1, h2)
        new_w = w1 + w2 + padding
        new_img = Image.new('RGB', (new_w, new_h), color=bg_color)
        y1 = (new_h - h1) // 2
        y2 = (new_h - h2) // 2
        new_img.paste(im1, (0, y1))
        new_img.paste(im2, (w1 + padding, y2))
    elif direction == 'vertical':
        new_w = max(w1, w2)
        new_h = h1 + h2 + padding
        new_img = Image.new('RGB', (new_w, new_h), color=bg_color)
        x1 = (new_w - w1) // 2
        x2 = (new_w - w2) // 2
        new_img.paste(im1, (x1, 0))
        new_img.paste(im2, (x2, h1 + padding))
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'")

    if save_path:
        new_img.save(save_path)

    return new_img

# register_heif_opener()
def convert_heic_to_jpeg(heic_path, jpeg_path, img_size = (640, 480)):
    try:
        # Open the HEIC image
        img = Image.open(heic_path)

        # Convert to RGB mode if necessary
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        
        img = img.resize(img_size)
        
        # Save the image in JPEG format
        img.save(jpeg_path, "JPEG", quality=95)

        print(f"Successfully converted {heic_path} to {jpeg_path}")

    except Exception as e:
        print(f"Error converting {heic_path}: {e}")

def add_text_2_img(img, text, font_size=40, xy=(20, 20), color=(0, 0, 255)):
    # 1. Create a drawing context
    draw = ImageDraw.Draw(img)

    # 2. Load a font (ensure 'arial.ttf' is available or use a full path)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        print("Using default font.")

    # 3. Add text
    draw.text(xy, text, fill=color, font=font) # Red color

    # 4. convert to bytes
    byte_io = io.BytesIO()
    img.save(byte_io, format='JPEG')
    jpeg_bytes = byte_io.getvalue()

    # 4. Save the result
    return jpeg_bytes

def draw_bbox(image_path, bboxs, labels, new_size = (1000, 600)):
    if image_path.startswith("http"):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    # Resize the image
    image = image.resize(new_size)

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box, label in zip(bboxs, labels):
        x_min, y_min, x_max, y_max = box
        width, height = x_max - x_min, y_max - y_min
        rect = patches.Rectangle((x_min, y_min),
            width,
            height,
            linewidth = 2,
            edgecolor = 'r',
            facecolor = 'none'
        )
        ax.add_patch(rect)
        plt.text(
            x_min, 
            y_min, 
            f"{label}", 
            color='white', 
            fontsize=12,
            bbox = dict(facecolor='red', alpha=0.5)
        )

    plt.axis('off')
    # plt.show()

def record_video_from_images(df, image_col_name,  fps = 30, output_path = './videos/monitor_video.mp4'):    
    frames = []
    for img_bytes in df[image_col_name]:
        # Use a library like OpenCV to decode the image bytes
        img_buf = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR) # Use IMREAD_COLOR for standard RGB
        if img is not None:
            frames.append(img)

    height, width, layers = frames[0].shape
    fps = fps # Desired frames per second
    video_filename = output_path

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    # Write the frames to the video file
    for frame in frames:
        out.write(frame)

    # Release the VideoWriter
    out.release()
    logging.info(f"Successfully created video: {video_filename}")

def _extract_video_frames_with_ffmpeg(video_path, out_fps, duration=-1):
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise IOError(
            f"Cannot decode video with OpenCV and ffmpeg is not installed: {video_path}"
        )

    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-hwaccel",
        "none",
    ]

    if duration != -1:
        command.extend(["-sseof", f"-{duration}"])

    command.extend([
        "-i",
        video_path,
        "-vf",
        f"fps={out_fps}",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-",
    ])

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise IOError(f"ffmpeg failed to extract frames from {video_path}") from exc

    images = []
    buffer = b""
    start_marker = b"\xff\xd8"
    end_marker = b"\xff\xd9"

    while True:
        chunk = process.stdout.read(1024 * 1024)
        if not chunk:
            break
        buffer += chunk

        while True:
            start_idx = buffer.find(start_marker)
            if start_idx == -1:
                buffer = buffer[-1:] if buffer else b""
                break

            end_idx = buffer.find(end_marker, start_idx + 2)
            if end_idx == -1:
                buffer = buffer[start_idx:]
                break

            jpeg_bytes = buffer[start_idx:end_idx + 2]
            buffer = buffer[end_idx + 2:]
            image = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
            images.append(image.copy())

    stderr_output = process.stderr.read().decode("utf-8", errors="ignore")
    return_code = process.wait()
    if return_code != 0:
        raise IOError(
            f"ffmpeg failed to extract frames from {video_path}: {stderr_output.strip()}"
        )

    return images

def convert_video_to_images(video_path, out_fps=30, duration=1):
    """
    Extracts frames from a video and returns them as a list of PIL images.

    Args:
        video_path (str): Path to the input video file.
        out_fps (float): Number of frames per second to extract (default: 30).
        duration (float): Number of seconds from the end of the video to extract.
            Use -1 to extract the complete video. Default: 1.
    Returns:
        list[PIL.Image.Image]: Extracted images in RGB format.
    """
    if out_fps <= 0:
        raise ValueError("out_fps must be > 0")
    if duration != -1 and duration <= 0:
        raise ValueError("duration must be > 0 or -1")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is not None:
        return _extract_video_frames_with_ffmpeg(
            video_path,
            out_fps,
            duration=duration,
        )

    cap = _open_video_capture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if video_fps <= 0 or frame_count <= 0:
        if duration != -1:
            cap.release()
            raise IOError(
                "Video metadata is unavailable and ffmpeg is not installed, "
                f"so duration-based extraction cannot be applied: {video_path}"
            )

        logging.warning(
            "Video FPS or frame count not available; extracting every frame until EOF"
        )
        frame_interval = 1
    else:
        frame_interval = max(1, int(round(video_fps / float(out_fps))))

        if duration != -1:
            video_duration = frame_count / float(video_fps)
            start_seconds = max(0.0, video_duration - duration)
            if start_seconds > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000.0)

    saved = 0
    idx = 0
    images = []

    while True:
        with _suppress_stderr(), _suppress_cv2_logging():
            ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb_frame))
            saved += 1
        idx += 1

    cap.release()
    if saved == 0:
        raise IOError(f"No frames decoded from video file: {video_path}")

    logging.info(f"Extracted {saved} frames from {video_path}")
    return images

if __name__ == "__main__":
    root_dir = "/home/yang/datasets/visual_image"
    path_list = create_file_list(root_dir)

    for path in path_list:
        img = process_image(path)
        img.save("edited_" + os.path.basename(path))
