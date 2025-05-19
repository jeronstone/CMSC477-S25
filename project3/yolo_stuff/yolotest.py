from ultralytics import YOLO
import cv2
import time
from robomaster import robot
from robomaster import camera
from queue import Empty

print('model')
model = YOLO(r"C:\Users\jesto\Desktop\CMSC477\CMSC477-S25\runs\detect\train5\weights\best.pt")

ep_robot = robot.Robot()
ep_robot.initialize(conn_type="sta", sn="3JKCH7T001008H")
ep_camera = ep_robot.camera
ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

while True:
    try:
        frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
    except Empty:
        time.sleep(0.001)
        continue
    
    if frame is not None:
        start = time.time()
        if model.predictor:
            model.predictor.args.verbose = False
        result = model.predict(source=frame, show=False)[0]


        # DIY visualization is much faster than show=True for some reason
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy.cpu().numpy().flatten()
            cls = int(box.cls)
            class_label = result.names[cls]
            cv2.rectangle(frame,
                          (int(xyxy[0]), int(xyxy[1])), 
                          (int(xyxy[2]), int(xyxy[3])),
                           color=(0, 0, 255), thickness=2)
            
            cv2.putText(frame, class_label, (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


        # print(results)


        end = time.time()
        print(1.0 / (end-start))
