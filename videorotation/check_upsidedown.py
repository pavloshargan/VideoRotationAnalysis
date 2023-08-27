import os
import cv2
import numpy as np
import tensorflow as tf
import subprocess
import time
import tempfile
import pkg_resources

class VideoRotationAnalysis:
    MODEL_PATH = pkg_resources.resource_filename('videorotation', 'mobilenet2_bi-rotnet.tflite')

    def __init__(self, frames_per_video=12, frames_threshold=8):
        self.interpreter = tf.lite.Interpreter(model_path=self.MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.frames_per_video = frames_per_video
        self.frames_threshold = frames_threshold

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Unload the model by setting it to None
        self.interpreter = None
        self.input_details = None
        self.output_details = None

    @staticmethod
    def cmd(command):
        response = (
            subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            .stdout.read()
            .decode(errors="ignore")
        )
        return response


    def extract_and_save_frames_for_video(self, video, temp_dir):
        frame_paths = []
        duration = float(VideoRotationAnalysis.cmd(f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {video}").strip())
        step = duration / (self.frames_per_video+1)
        for i in range(self.frames_per_video):
            time_stamp = i * step
            frame_path = os.path.join(temp_dir, f"{os.path.basename(video)}_frame_{i}.jpg")
            res = VideoRotationAnalysis.cmd(f"ffmpeg -ss {time_stamp} -i {video} -vframes 1 {frame_path}")
            frame_paths.append(frame_path)
        return frame_paths

    def is_flipped(self, image_path):
        # Open and preprocess the image using OpenCV
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        image = cv2.resize(image, (256, 256))
        image = image / 255.0  # Normalizing to [0,1]
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)

        # Make prediction
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Assuming label 1 indicates the flipped class
        return bool(np.argmax(prediction))

    def check_if_upsidedown_for_video(self, video):
        with tempfile.TemporaryDirectory() as temp_dir:
            time_s = time.time()
            frame_paths = self.extract_and_save_frames_for_video(video, temp_dir)
            print("extract: ", time.time()-time_s)
            time_s = time.time()
            result = []
            for path in frame_paths:
                flipped = self.is_flipped(path)
                result.append(flipped)
            print("predict: ", time.time()-time_s)

            return sum(1 for x in result if x) >= self.frames_threshold