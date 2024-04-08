import cv2
import os
import tkinter as tk
from tkinter import filedialog

class PotholeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pothole Detection App")

        self.btn_browse = tk.Button(root, text="Browse Image", command=self.browse_image)
        self.btn_browse.pack(pady=10)

        self.btn_detect = tk.Button(root, text="Detect Potholes", command=self.detect_potholes)
        self.btn_detect.pack(pady=10)

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
        if file_path:
            self.image_path = file_path

    def detect_potholes(self):
        try:
            img = cv2.imread(self.image_path)

            # Your existing detection code
            net = cv2.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
            model = cv2.dnn_DetectionModel(net)
            model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
            classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)

            for (classId, score, box) in zip(classIds, scores, boxes):
                cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                              color=(0, 255, 0), thickness=2)

            cv2.imshow("Pothole", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except AttributeError:
            print("Please select an image first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = PotholeDetectionApp(root)
    root.mainloop()
