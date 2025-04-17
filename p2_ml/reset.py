from robomaster import robot
import time

ep_robot = robot.Robot()
ep_robot.initialize(conn_type="sta", sn="3JKCH7T001008H")
ep_chassis = ep_robot.chassis
ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
ep_arm = ep_robot.robotic_arm
ep_gripper = ep_robot.gripper

ep_gripper.open(power=150)
time.sleep(1.0)
ep_gripper.pause()

ep_arm.moveto(x=200, y=-50).wait_for_completed()
time.sleep(1.0)

ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
ep_robot.close()