import cv2
import numpy as np
from ultralytics import YOLO

class Vision():
    
    def __init__(self, model_pth):
        self.model = YOLO(model_pth)
    
    '''
    @param frame: camera frame to analyze
    @param hough (default False): true if perform hough on detections
    @return 3 tuple corners, depth, hough (none if @param hough=False)
    '''
    def get_yolo_pred(self, frame, hough=False):

        clean_frame = frame.copy()
        
        if self.model.predictor:
            self.model.predictor.args.verbose = False
        result = self.model.predict(source=frame, show=False)[0]
        
        detections = []
        
        boxes = result.boxes
        if len(boxes) != 0:
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

                cv2.putText(frame, str((round(corners[2], 2), round(corners[1], 2))), (int(xyxy[2]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
                cv2.putText(frame, str((round(corners[0], 2), round(corners[3], 2))), (int(xyxy[0]), int(xyxy[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
                cv2.putText(frame, str((round(corners[2], 2), round(corners[3], 2))), (int(xyxy[2]), int(xyxy[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
                cv2.putText(frame, str((round(corners[0], 2), round(corners[1], 2))), (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
            
                cls = int(box.cls)
            
                if hough:
                    if clean_frame is not None:
                        detection_y = int(xyxy[3]) - int(xyxy[1])
                        detected_block = clean_frame[(int(xyxy[1])+int(detection_y*1/4)):(int(xyxy[3])-int(detection_y*1/4)), int(xyxy[0]):int(xyxy[2])]
                        detected_block_gray = cv2.cvtColor(detected_block, cv2.COLOR_BGR2GRAY)
                        detected_block_gray_gaussian = cv2.GaussianBlur(detected_block_gray, (3, 3), 0)
                        detected_block_lines = cv2.Canny(detected_block_gray_gaussian, 100, 250, None, 3)

                    # detected_block_lines_hough = cv2.HoughLines(detected_block_lines, 1, np.pi / 180, 30, None, 0, 0)
                    detected_block_lines_hough = cv2.HoughLinesP(detected_block_lines, 1, np.pi / 180, 25, None, 20, 1)
                    # print(detected_block_lines_hough) 
                
                    depth = (0.064*314.0)/(int(xyxy[2])-int(xyxy[0])) # actual block length is 0.158 meters; however, in the worst case, we will only see around 0.1 meters of the block, so use that as the depth. this means we underestimate the depth at every iteration

                    detections.append((cls, corners, depth, detected_block_lines_hough))
                else:
                    depth = (0.2*314.0)/(int(xyxy[2])-int(xyxy[0]))
                    detections.append((cls, corners, depth, None))
                
        return frame, detections
    
    # for getting hough within bounding box, use get_yolo_pred with hough=True
    def get_hough_entire_frame(self, frame):
        detected_block_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_block_gray_gaussian = cv2.GaussianBlur(detected_block_gray, (3, 3), 0)
        detected_block_lines = cv2.Canny(detected_block_gray_gaussian, 100, 250, None, 3)

        # detected_block_lines_hough = cv2.HoughLines(detected_block_lines, 1, np.pi / 180, 30, None, 0, 0)
        detected_block_lines_hough = cv2.HoughLinesP(detected_block_lines, 1, np.pi / 180, 25, None, 20, 1)
        # print(detected_block_lines_hough)
        
        return detected_block_lines_hough