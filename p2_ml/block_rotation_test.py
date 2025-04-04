from ultralytics import YOLO
import cv2

print('model')
model = YOLO(r"..\runs\detect\train5\weights\best.pt")

frame = cv2.imread(r"train_images\old\block_in_frame_37.png") # 0; 37; 

if frame is not None:
    result = model.predict(source=frame, show=False)[0]

    boxes = result.boxes
    for box in boxes:
        xyxy = box.xyxy.cpu().numpy().flatten()
        cv2.rectangle(frame,
                    (int(xyxy[0]), int(xyxy[1])), 
                    (int(xyxy[2]), int(xyxy[3])),
                    color=(0, 0, 255), thickness=2)
        
        corners = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
        corners[0] = 2*((corners[0]) / 640) - 1
        corners[2] = 2*((corners[2]) / 640) - 1
        
        corners[1] = 2*((corners[1]) / 360) - 1
        corners[3] = 2*((corners[3]) / 360) - 1

        cv2.putText(frame, str((round(corners[0], 2), round(corners[1], 2))), (int(xyxy[0]) - 50, int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
        cv2.putText(frame, str((round(corners[2], 2), round(corners[1], 2))), (int(xyxy[2]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
        cv2.putText(frame, str((round(corners[0], 2), round(corners[3], 2))), (int(xyxy[0]) - 50, int(xyxy[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
        cv2.putText(frame, str((round(corners[2], 2), round(corners[3], 2))), (int(xyxy[2]), int(xyxy[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))

        detected_block = frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
        detected_block_gray = cv2.cvtColor(detected_block, cv2.COLOR_BGR2GRAY)
        # detected_block_gray_gaussian = cv2.GaussianBlur(detected_block_gray, (3, 3), 0)
        detected_block_lines = cv2.Canny(detected_block_gray, 100, 250, None, 3)

        cv2.imshow('frame', detected_block_lines)
        key = cv2.waitKey(0)