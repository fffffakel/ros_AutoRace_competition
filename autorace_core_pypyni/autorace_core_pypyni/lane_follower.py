import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

class LaneFollower(Node):
    def __init__(self):
        super().__init__('lane_follower')
        
        qos_camera = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_command = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)

        self.sub_camera = self.create_subscription(Image, '/color/image', self.camera_callback, qos_camera)
        self.sub_command = self.create_subscription(String, '/course_code', self.command_callback, qos_command)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.bridge = CvBridge()
        self.twist = Twist()
        self.stop_robot = True 

        # === PID (Чуть агрессивнее) ===
        self.Kp = 0.005 
        self.Ki = 0.0000
        self.Kd = 0.005
        
        self.desiredV = 0.15 # Средняя скорость
        self.E = [0] * 15
        self.old_e = 0
        self.lane_width_px = 300
        self.mode = 'center'

        self.get_logger().info("Lane Follower: BACK TO BASICS (Split Screen + Small Offset)")

    def command_callback(self, msg):
        cmd = msg.data.lower().strip()
        if cmd in ['left', 'right', 'center']:
            self.mode = cmd
            self.stop_robot = False
            self.get_logger().info(f"👉 MODE: {cmd.upper()}")
        elif cmd == 'stop':
            self.stop_robot = True
        elif cmd == 'go':
            self.stop_robot = False

    def camera_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: return

        height, width, _ = cv_image.shape
        crop_img = cv_image[int(height*0.7):height, 0:width]
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        
        lower_yellow = np.array([20, 80, 80]); upper_yellow = np.array([40, 255, 255])
        lower_white = np.array([0, 0, 180]); upper_white = np.array([180, 50, 255])

        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        h_crop, w_crop = mask_yellow.shape
        mid = int(w_crop / 2)
        
        mask_yellow[:, mid:] = 0 
        mask_white[:, :mid] = 0   

        M_y = cv2.moments(mask_yellow)
        cy = int(M_y['m10']/M_y['m00']) if M_y['m00'] > 0 else None

        M_w = cv2.moments(mask_white)
        cw = int(M_w['m10']/M_w['m00']) if M_w['m00'] > 0 else None

        if cy and cw: self.lane_width_px = cw - cy

        target_center = width / 2 
        
        # === ИСПРАВЛЕНИЕ ЗДЕСЬ ===
        # Увеличим смещение, чтобы робот увереннее сворачивал
        safety_offset = 40
        
        if self.mode == 'right':
            # Едем НАПРАВО -> Смещаем цель ВПРАВО (+ offset)
            # Чтобы робот прижался к Белой полосе
            if cw: 
                target_center = cw - (self.lane_width_px / 2) + safety_offset
            elif cy:
                target_center = cy + (self.lane_width_px / 2) + safety_offset
            else:
                target_center = width 
        
        elif self.mode == 'left':
            # Едем НАЛЕВО -> Смещаем цель ВЛЕВО (- offset)
            # Чтобы робот прижался к Желтой полосе
            if cy: 
                target_center = cy + (self.lane_width_px / 2) - safety_offset
            elif cw:
                target_center = cw - (self.lane_width_px / 2) - safety_offset
            else:
                target_center = 0 

        else: # CENTER
            if cy and cw: target_center = (cy + cw) / 2
            elif cy: target_center = cy + (self.lane_width_px / 2)
            elif cw: target_center = cw - (self.lane_width_px / 2)

        error = (width / 2) - target_center

        # Debug visualization
        debug_frame = crop_img.copy()
        if cy: cv2.circle(debug_frame, (cy, 20), 8, (0, 255, 255), -1)
        if cw: cv2.circle(debug_frame, (cw, 20), 8, (255, 255, 255), -1)
        cv2.circle(debug_frame, (int(target_center), 40), 6, (0, 255, 0), -1)
        # Показываем текст режима
        cv2.putText(debug_frame, f"Mode: {self.mode}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.imshow("Simple Lane Tracking", debug_frame)
        cv2.waitKey(1)
        
        self.pid_control(error)

    def pid_control(self, error):
        if self.stop_robot:
            self.twist.linear.x = 0.0; self.twist.angular.z = 0.0
            self.pub_cmd_vel.publish(self.twist); return
        
        e_P = error
        self.E.pop(0); self.E.append(error)
        w = self.Kp * e_P + self.Ki * sum(self.E) + self.Kd * (error - self.old_e)
        w = max(min(w, 2.0), -2.0)
        
        linear_v = self.desiredV * (1 - 0.5 * abs(w) / 2.0)
        if linear_v < 0.05: linear_v = 0.05

        self.twist.linear.x = linear_v
        self.twist.angular.z = float(w) 
        self.pub_cmd_vel.publish(self.twist)
        self.old_e = error

def main(args=None):
    rclpy.init(args=args)
    node = LaneFollower()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown(); cv2.destroyAllWindows()
if __name__ == '__main__': main()