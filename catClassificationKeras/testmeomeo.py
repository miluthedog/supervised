import cv2 as cv
import numpy as np
from keras import models
import sys
sys.stdout.reconfigure(encoding='utf-8')


class testmeomeo:
    def __init__(self, model_path, camera_index, threshold):
        self.model = models.load_model(model_path)
        self.camera_index = camera_index
        self.threshold = threshold
        self.testing()
    
    def testing(self):
        cam = cv.VideoCapture(self.camera_index)

        while True:
            _, frame = cam.read()

            resized_frame = cv.resize(frame, (128, 128)) / 255.0
            resized_frame = np.expand_dims(resized_frame, axis=0)

            prediction = self.model.predict(resized_frame)
            confidence = prediction[0][0] * 100
            label = "meo meo" if prediction[0] > self.threshold else "???"
            cv.putText(frame, f"{label} ({confidence:.2f}%)", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv.imshow('Meo meo finder', frame)

            if cv.waitKey(1) == 27:
                break
        
        cam.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    testmeomeo('1catClassification/meomeo.keras', 0, 0.5)

# input:
    # model_path: path to model
    # camera_index: default camera = 0
    # threshold: confidence required