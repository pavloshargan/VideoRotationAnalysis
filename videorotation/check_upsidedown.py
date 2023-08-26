import os
import cv2
import numpy as np
import tensorflow as tf
import subprocess
import time
import tempfile
import pkg_resources

class VideoRotationAnalysis:

    MODEL_PATH = pkg_resources.resource_filename('videorotation', 'rotnet_street_view_resnet50_keras2_pb')

    def __init__(self, frames_per_video=12):
        self.model_location = tf.saved_model.load(self.MODEL_PATH)
        self.model = self.model_location.signatures["serving_default"]
        self.frames_per_video = frames_per_video

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Unload the model by setting it to None
        self.model = None

    @staticmethod
    def cmd(command):
        response = (
            subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            .stdout.read()
            .decode(errors="ignore")
        )
        return response

    @staticmethod
    def keras_imagenet_caffe_preprocess(x):
        if not issubclass(x.dtype.type, np.floating):
            x = x.astype(np.float32, copy=False)

        x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]

        for i in range(3):
            x[..., i] -= mean[i]

        return x

    def preprocess_image(self, image_path, target_size=(224, 224)):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)
        image = np.expand_dims(image, axis=0)
        return self.keras_imagenet_caffe_preprocess(image)

    def predict_angle(self, paths):
        result = []
        for path in paths:
            image_tensor = self.preprocess_image(path)
            output = self.model(tf.convert_to_tensor(image_tensor))

            if isinstance(output, dict):
                batch_predictions = output['fc360'].numpy()
            else:
                batch_predictions = output.numpy()

            predicted_angle = np.argmax(batch_predictions, axis=1)[0]
            result.append(predicted_angle)

        return result

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

    def is_mostly_upside_down(self, angles):
        count_in_range = sum(1 for angle in angles if 150 <= angle <= 210)
        return count_in_range >= (self.frames_per_video//3)

    def check_if_upsidedown_for_video(self, video):
        with tempfile.TemporaryDirectory() as temp_dir:
            time_s = time.time()
            frame_paths = self.extract_and_save_frames_for_video(video, temp_dir)
            print("extract: ", time.time()-time_s)
            time_s = time.time()
            predicted_angles = self.predict_angle(frame_paths)
            print("predict: ", time.time()-time_s)
            return self.is_mostly_upside_down(predicted_angles), predicted_angles