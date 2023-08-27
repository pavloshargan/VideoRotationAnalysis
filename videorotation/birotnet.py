import numpy as np
import tensorflow as tf
import cv2

class ImageFlippedPredictor:
    MODEL_PATH = "mobilenet2_bi-rotnet.tflite"

    def __init__(self):
        self.interpreter = None

    def __enter__(self):
        self.interpreter = tf.lite.Interpreter(model_path=self.MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.interpreter = None
        self.input_details = None
        self.output_details = None

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

if __name__ == "__main__":
    # Example of how to use the class with 'with' block:
    with ImageFlippedPredictor() as predictor:
        from glob import glob
        for image_path in glob(r'C:\Users\Pavlo\Desktop\bi-rotnet-test\*'):
            print(image_path)
            print(predictor.is_flipped(image_path))  # This will print True or False
