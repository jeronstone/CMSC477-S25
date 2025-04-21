from robomaster import robot

def sp_callback(pos):
    x,y,z = pos
    print(f'{x},{y},{z}')

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH7T001008H")

    ep_chassis = ep_robot.chassis

    x_val = 1.0
    y_val = 1.0
    
    ep_chassis.sub_position(cs=1, freq=5, callback=sp_callback)

    # Forward 0.5 meters
    ep_chassis.move(x=x_val, y=0, z=0, xy_speed=0.5).wait_for_completed()

    # Backward 0.5 meters
    ep_chassis.move(x=-x_val, y=0, z=0, xy_speed=0.5).wait_for_completed()

    # Move 0.5 meters left
    ep_chassis.move(x=0, y=-y_val, z=0, xy_speed=0.5).wait_for_completed()

    # Move 0.5 meters right
    ep_chassis.move(x=0, y=y_val, z=0, xy_speed=0.5).wait_for_completed()

    ep_chassis.unsub_position()

    ep_robot.close()
