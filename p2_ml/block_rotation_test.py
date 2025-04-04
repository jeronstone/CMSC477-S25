from ultralytics import YOLO
import cv2
import numpy as np
import math

print('model')
model = YOLO(r"..\runs\detect\train\weights\best.pt")

for i in range(1, 133):

    frame = cv2.imread(f"p2_ml\\cmsc477_yolo\\datasets\\train\\block_{i}.png")

    if frame is not None:
        clean_frame = frame.copy()

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

            detected_block = clean_frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
            detected_block_gray = cv2.cvtColor(detected_block, cv2.COLOR_BGR2GRAY)
            detected_block_gray_gaussian = cv2.GaussianBlur(detected_block_gray, (3, 3), 0)
            detected_block_lines = cv2.Canny(detected_block_gray_gaussian, 100, 250, None, 3)

            # detected_block_lines_hough = cv2.HoughLines(detected_block_lines, 1, np.pi / 180, 30, None, 0, 0)
            detected_block_lines_hough = cv2.HoughLinesP(detected_block_lines, 1, np.pi / 180, 25, None, 15, 1)
            # print(detected_block_lines_hough)

            if detected_block_lines_hough is not None:
                most_vertical = detected_block_lines_hough[0][0]
                most_horizontal = detected_block_lines_hough[0][0]
                for i in range(len(detected_block_lines_hough)):
                    # rho = detected_block_lines_hough[i][0][0]
                    # theta = detected_block_lines_hough[i][0][1]
                    # a = math.cos(theta)
                    # b = math.sin(theta)
                    # x0 = a * rho
                    # y0 = b * rho
                    # pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                    # pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                    # cv2.line(detected_block, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
                    l = detected_block_lines_hough[i][0]
                    if abs(l[2] - l[0]) < abs(most_vertical[2] - most_vertical[0]):
                        most_vertical = l
                    if abs(l[3] - l[1]) < abs(most_horizontal[3] - most_horizontal[1]):
                        most_horizontal = l
                # cv2.line(detected_block, (most_vertical[0], most_vertical[1]), (most_vertical[2], most_vertical[3]), (0,0,0), 3, cv2.LINE_AA)
                cv2.line(detected_block, (most_horizontal[0], most_horizontal[1]), (most_horizontal[2], most_horizontal[3]), (255,255,255), 3, cv2.LINE_AA)

                if most_horizontal is not most_vertical:
                    # most_vertical_angle = math.atan2(most_vertical[3] - most_vertical[1], most_vertical[2] - most_vertical[0])
                    most_horizontal_angle = math.atan2(most_horizontal[3] - most_horizontal[1], most_horizontal[2] - most_horizontal[0])
                    # print(f"vertical {most_vertical_angle} horizontal {most_horizontal_angle}")
                    print(f"horizontal {most_horizontal_angle}; rotation to align: {most_horizontal_angle}")
                

            cv2.imshow('frame', detected_block)
            key = cv2.waitKey(0)
            if key == ord('q'):
                exit()
            cv2.destroyWindow('frame')