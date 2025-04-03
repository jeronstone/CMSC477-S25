import cv2
from robomaster import robot
from robomaster import camera

i=0

def save_image(image, filename="block"):
    global i
    cv2.imwrite(f'train_images_v2/{filename}_{i}.png', image)
    i+=1
    print(f"Image saved")

def on_button_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        save_image(param)

# Use vid instead of ep_camera to use your laptop's webcam
# vid = cv2.VideoCapture(0)


ep_robot = robot.Robot()
ep_robot.initialize(conn_type="sta", sn="3JKCH8800100YR")
ep_camera = ep_robot.camera
ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

cv2.namedWindow("Camera Feed")

while True:
    # ret, frame = vid.read()
    frame = ep_camera.read_cv2_image()
    if frame is not None:
        cv2.setMouseCallback("Camera Feed", on_button_click, frame)
        cv2.imshow("Camera Feed", frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break



