import cv2
import pandas as pd
from PIL import Image
import io
import matplotlib.pyplot as plt


# new_row_data = {'A': 5, 'B': 6}

# # Convert new data to a DataFrame
# # You can also pass a list of dicts to the DataFrame constructor
# new_row_df = pd.DataFrame([new_row_data])

# # Concatenate to create a new DataFrame
# df = pd.concat([df, new_row_df], ignore_index=True)

def read_video_with_timestamps(video_path, save_dir = None, trg_img_size = (848, 480), save_image_iter = 30):
    """
    Reads frames and calculates timestamps for an MP4 video using OpenCV.

    Args:
        video_path (str): The path to the MP4 video file.
    """
    column_names = ['timestamp_sec', 'image']
    df = pd.DataFrame(columns=column_names)

    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get the frames per second (FPS) of the video
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Could not get FPS, default to 30.")
        fps = 30 # Fallback default FPS
    
    count = 0
    success = True
    print(f"Video opened successfully with FPS: {fps}")

    while success:
        success, image = vidcap.read()
        if success:
            # Calculate the timestamp in seconds
            timestamp_seconds = count / fps

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # start_col, end_col = 320, 1760
            # image = image[:, start_col:end_col, :]
            
            
            # Create a BytesIO object to hold the image data
            new_dims = trg_img_size
            image = cv2.resize(image, new_dims)
            # img_byte_arr = image.tobytes()
            if count == 500:
                plt.imshow(image)
            # Convert the NumPy array to a PIL Image
            pil_image = Image.fromarray(image)

            # Use BytesIO to save the image to an in-memory buffer in JPEG format
            byte_io = io.BytesIO()
            pil_image.save(byte_io, format='JPEG')
            if save_dir is not None and count % save_image_iter == 0:
                pil_image.save(save_dir + f"/frame_{count}.jpg", format='JPEG')

            # Get the JPEG-encoded bytes
            jpeg_bytes = byte_io.getvalue()
            print(jpeg_bytes[:20])
            df_row = {'timestamp_sec': timestamp_seconds, 'image': jpeg_bytes}
            df = pd.concat([df, pd.DataFrame([df_row])], ignore_index=True)
            
            count += 1
        else:
            break

    vidcap.release()
    # cv2.destroyAllWindows() # Uncomment if using cv2.imshow
    print(f"Finished reading {count} frames.")
    return df

# Example usage:
# Replace 'your_video.mp4' with the actual path to your video file
if __name__ == "__main__":
    df = read_video_with_timestamps('./videos/port_0001.mp4', '/home/yang/datasets/white_board_image3', save_image_iter = 10000)
    print(df.head())
    df.to_parquet('./videos/port_0001.parquet', engine='fastparquet')
    print("successfully saved to parquet file.")
    plt.show() # Displays the figure