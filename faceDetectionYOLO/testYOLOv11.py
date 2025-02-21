from ultralytics import YOLO
import cv2 as cv
import sys
sys.stdout.reconfigure(encoding='utf-8')

class testYOLO:
    def __init__(self, model_path, camera_index, threshold):
        self.model = YOLO(model_path)
        self.camera_index = camera_index
        self.threshold = threshold
        self.testing()

    def testing(self):
        cam = cv.VideoCapture(self.camera_index)

        while True:
            _, frame = cam.read()
            results = self.model(frame)

            for result in results:
                for box in result.boxes:
                    if box.conf[0] > self.threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        class_name = self.model.names[cls]

                        cv.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv.putText(frame, f'{class_name} {box.conf[0]:.2f}', 
                                   (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 
                                   0.5, (255, 255, 0), 2)

            cv.imshow('Face Tracking', frame)

            if cv.waitKey(1) == 27:
                break

        cam.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    testYOLO('faceDetectionYOLO/best.pt', 0, 0.7)

# input:
    # model_path: path to model
    # camera_index: default camera = 0
    # threshold: confidence required