import cv2 as cv
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import time
import geocoder
import os
import random

class_name = []
with open(os.path.join("project_files", 'obj.names'), 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

net1 = cv.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

result_path = "pothole_coordinates"
g = geocoder.ip('me')
starting_time = time.time()
Conf_threshold = 0.5
NMS_threshold = 0.4
frame_counter = 0
i = 0
b = 0

class PotholeDetectorApp:
    def __init__(self, root, video_source):
        self.pothole_count = 0
        self.root = root
        self.root.title("Pothole Detector App")
        self.video_source = video_source
        self.cap = cv.VideoCapture(self.video_source)
        self.canvas = tk.Canvas(root, width=self.cap.get(3), height=self.cap.get(4))
        self.canvas.pack()
        self.lbl_heading = tk.Label(root, text="Pothole Detection", font=("Arial", 16, "bold"))
        self.lbl_heading.pack(pady=20)
        self.lbl_heading.place(relx=0.5,rely=0.75,anchor='center')
        self.btn_browse = tk.Button(root, text="Browse Video", command=self.browse_video, bg="blue", fg="white", font=("Arial", 12))
        self.btn_browse.pack(pady=10)
        self.btn_browse.place(relx=0.5, rely=0.8, anchor="center")
        self.btn_start = tk.Button(root, text="Start Detection", command=self.start_detection, bg="green", fg="white", font=("Arial", 12))
        self.btn_start.pack(pady=10)
        self.btn_start.place(relx=0.5, rely=0.85, anchor="center")
        self.btn_stop = tk.Button(root, text="Stop Detection", command=self.stop_detection, bg="red", fg="white", font=("Arial", 12))
        self.btn_stop.pack(pady=10)
        self.btn_stop.place(relx=0.5, rely=0.9, anchor="center")
        self.is_detecting = False
        self.result = None
        self.width = self.cap.get(3)  # Get width of the video
        self.height = self.cap.get(4)  # Get height of the video

    def browse_video(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.video_source = file_path
            self.cap = cv.VideoCapture(self.video_source)
            messagebox.showinfo("Video Uploaded", "Video has been uploaded successfully!")

    def start_detection(self):
        self.is_detecting = True
        self.pothole_count=0
        self.result = cv.VideoWriter('result.avi', cv.VideoWriter_fourcc(*'MJPG'), 10, (int(self.width), int(self.height)))
        self.detect_potholes()

    def stop_detection(self):
        self.is_detecting = False
        if self.result:
            self.result.release()

        # Calculate road quality here
        if self.pothole_count<=3:
            road_quality=5
        elif self.pothole_count<=6:
            road_quality=4
        elif self.pothole_count<=9:
            road_quality=3
        elif self.pothole_count<=12:
            road_quality=2
        else:
            road_quality=1
        # road_quality = random.randint(3, 5)  # Example: Random road quality
        self.show_road_quality_popup(road_quality)

    def show_road_quality_popup(self,road_quality):
        messagebox.showinfo("Road Quality", f"The road quality is {road_quality}")

    def detect_potholes(self):
        global frame_counter, i, b
        while self.is_detecting:
            ret, frame = self.cap.read()
            frame_counter += 1
            if ret:
                if ret == False:
                    break
                classes, scores, boxes = model1.detect(frame, Conf_threshold, NMS_threshold)
                for (classid, score, box) in zip(classes, scores, boxes):
                    label = "pothole"
                    x, y, w, h = box
                    recarea = w * h
                    area = self.width * self.height
                    if (len(scores) != 0 and scores[0] >= 0.7):
                        # cv.putText(frame, f'Road_Quality: {random.randint(3,5)}', (100, 100), cv.FONT_HERSHEY_COMPLEX,
                        #         0.7, (0, 255, 0), 2)
                        if ((recarea / area) <= 0.1 and box[1] < 600):
                            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                            cv.putText(frame, "%" + str(round(scores[0] * 100, 2)) + " " + label,
                                    (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                            if (i == 0):
                                cv.imwrite(os.path.join(result_path, 'pothole' + str(i) + '.jpg'), frame)
                                with open(os.path.join(result_path, 'pothole' + str(i) + '.txt'), 'w') as f:
                                    f.write(str(g.latlng))
                                i = i + 1
                                self.pothole_count+=1
                            if (i != 0):
                                if ((time.time() - b) >= 2):
                                    cv.imwrite(os.path.join(result_path, 'pothole' + str(i) + '.jpg'), frame)
                                    with open(os.path.join(result_path, 'pothole' + str(i) + '.txt'), 'w') as f:
                                        f.write(str(g.latlng))
                                    b = time.time()
                                    i = i + 1
                                    self.pothole_count+=1

                endingTime = time.time() - starting_time
                fps = frame_counter / endingTime
                cv.putText(frame, f'FPS: {fps}', (20, 50), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                cv.putText(frame, f'Pothole Count: {self.pothole_count}', (20, 80), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                # cv.imshow('frame', frame)
                if self.result:
                    self.result.write(frame)
                self.display_frame(frame)
                key = cv.waitKey(1)
                if key == ord('q'):
                    self.stop_detection()
                self.root.update()

        self.cap.release()
        cv.destroyAllWindows()

    def display_frame(self, frame):
    # Resize the frame to fit the canvas while maintaining the aspect ratio
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        # Calculate the aspect ratios
        aspect_ratio_canvas = canvas_width / canvas_height
        aspect_ratio_frame = frame_width / frame_height

        # Calculate the new width and height for resizing
        if aspect_ratio_frame > aspect_ratio_canvas:
            new_width = canvas_width
            new_height = int(canvas_width / aspect_ratio_frame)
        else:
            new_width = int(canvas_height * aspect_ratio_frame)
            new_height = canvas_height

        # Resize the frame
        resized_frame = cv.resize(frame, (new_width, new_height))

        # Convert the resized frame to RGB format
        rgb_frame = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)

        # Convert the RGB frame to an ImageTk format
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the canvas with the resized frame
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk

    

if __name__ == "__main__":
    root = tk.Tk()
    app = PotholeDetectorApp(root, "test.mp4")
    root.mainloop()
