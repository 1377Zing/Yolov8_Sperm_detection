#-----------------------------------------------------------------------#
# predict.py integrates functions like single image prediction, camera detection, FPS testing, and directory traversal detection into one Python file. You can change the mode by specifying 'mode'.
#-----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    #----------------------------------------------------------------------------------------------------------#
    # mode specifies the test mode:
    # 'predict': Single image prediction. See detailed comments below for modifying the prediction process.
    # 'video': Video detection, can use camera or video. Check comments below for details.
    # 'fps': Test FPS using 'street.jpg' in 'img' folder. See comments below.
    # 'dir_predict': Traverse a folder for detection and save results. Defaults to 'img' folder and saves to 'img_out'.
    # 'heatmap': Visualize prediction results as a heatmap. See comments below.
    # 'export_onnx': Export the model to ONNX. Requires PyTorch 1.7.1 or higher.
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #-------------------------------------------------------------------------#
    # crop: Whether to crop the detected objects in single image prediction.
    # count: Whether to count the detected objects.
    # Only effective when mode='predict'.
    #-------------------------------------------------------------------------#
    crop = False
    count = False
    #----------------------------------------------------------------------------------------------------------#
    # video_path: Specifies the video path. Use 0 for camera.
    # video_save_path: Path to save the video. Leave empty to not save.
    # video_fps: FPS for the saved video.
    # Only effective when mode='video'.
    # Save the video by pressing Ctrl+C or reaching the last frame.
    #----------------------------------------------------------------------------------------------------------#
    video_path = 0
    video_save_path = ""
    video_fps = 25.0
    #----------------------------------------------------------------------------------------------------------#
    # test_interval: Number of image detections for FPS measurement. Larger value gives more accurate FPS.
    # fps_image_path: Image for FPS testing.
    # Only effective when mode='fps'.
    #----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path = ""
    #-------------------------------------------------------------------------#
    # dir_origin_path: Folder path for images to be detected.
    # dir_save_path: Folder path to save the detected images.
    # Only effective when mode='dir_predict'.
    #-------------------------------------------------------------------------#
    dir_origin_path = ""
    dir_save_path = ""
    #-------------------------------------------------------------------------#
    # heatmap_save_path: Path to save the heatmap. Defaults to 'model_data'.
    # Only effective when mode='heatmap'.
    #-------------------------------------------------------------------------#
    heatmap_save_path = "model_data/"
    #-------------------------------------------------------------------------#
    # simplify: Use Simplify ONNX.
    # onnx_save_path: Path to save the ONNX model.
    #-------------------------------------------------------------------------#
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    if mode == "predict":
        '''
        1. Save the detected image with r_image.save("img.jpg").
        2. Get bounding box coordinates in yolo.detect_image.
        3. Crop objects in yolo.detect_image using top, left, bottom, right.
        4. Write additional text on the predicted image in yolo.detect_image.
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop=crop, count=count)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Failed to read camera (video). Check camera installation or video path.")

        fps = 0.0
        while True:
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(yolo.detect_image(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print(f"fps= {fps:.2f}")
            frame = cv2.putText(frame, f"fps= {fps:.2f}", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print(f"Save processed video to the path: {video_save_path}")
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(f"{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1")

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(image, heatmap_save_path)

    elif mode == "export_onnx":
        yolo.convert_to_onnx(simplify, onnx_save_path)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")