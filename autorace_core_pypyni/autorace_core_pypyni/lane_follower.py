import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data # <--- ВАЖНО: Настройка QoS
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class LaneFollower(Node):
    def __init__(self):
        super().__init__('lane_follower')
        
        # --- ВАЖНО: ИСПОЛЬЗУЕМ qos_profile_sensor_data ---
        # Это позволяет принимать данные от камеры Gazebo, даже если она шлет "Best Effort"
        self.sub_camera = self.create_subscription(
            Image, 
            '/color/image', 
            self.camera_callback, 
            qos_profile_sensor_data
        )
        
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.bridge = CvBridge()
        self.twist = Twist()

        # PID Параметры
        self.Kp = 0.003
        self.Ki = 0.0001
        self.Kd = 0.004
        self.desiredV = 0.25
        
        self.old_e = 0
        self.E = [0] * 15
        self.last_cx_yellow = None
        self.last_cx_white = None
        
        self.get_logger().info("Lane Follower Node Started! Waiting for camera...")

    def camera_callback(self, msg):
        # Если мы тут, значит камера подключилась!
        # self.get_logger().info("Image received!") # Раскомментируй, если нужно проверить поток

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'CV Error: {e}')
            return

        height, width, _ = cv_image.shape
        
        # ОБРЕЗКА (ROI)
        crop_h_start = int(height * 0.60) 
        crop_img = cv_image[crop_h_start:height, 0:width]
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        
        # МАСКИ (Если робот не едет, проверь эти цвета калибровщиком!)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        
        lower_white = np.array([0, 0, 180]) 
        upper_white = np.array([180, 50, 255])

        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        cx_yellow = self.find_line_center(mask_yellow, is_yellow=True, width=width)
        cx_white = self.find_line_center(mask_white, is_yellow=False, width=width)

        target_center = (cx_yellow + cx_white) / 2
        robot_center = width / 2
        error = target_center - robot_center 

        self.calculate_pid(error)


    def find_line_center(self, mask, is_yellow, width):
        M = cv2.moments(mask)
        cx = None
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            if is_yellow: self.last_cx_yellow = cx
            else: self.last_cx_white = cx
        else:
            if is_yellow:
                cx = self.last_cx_yellow if self.last_cx_yellow is not None else 0
            else:
                cx = self.last_cx_white if self.last_cx_white is not None else width
        return cx

    def calculate_pid(self, error):
        e_P = error
        self.E.pop(0)
        self.E.append(error)
        e_I = sum(self.E)
        e_D = error - self.old_e
        
        w = self.Kp * e_P + self.Ki * e_I + self.Kd * e_D
        
        w_max = 2.5
        if w > w_max: w = w_max
        if w < -w_max: w = -w_max
        
        linear_v = self.desiredV * (1 - 0.5 * abs(w) / w_max)
        if linear_v < 0.05: linear_v = 0.05

        self.twist.linear.x = linear_v
        self.twist.angular.z = float(-w) 
        
        self.pub_cmd_vel.publish(self.twist)
        
        # ЛОГ: Показываем, что мы отправляем на колеса
        # Если тут нули - значит проблема в расчетах
        self.get_logger().info(f'CMD: Lin={linear_v:.2f}, Ang={w:.2f}, Err={error:.2f}')
        
        self.old_e = error

def main(args=None):
    rclpy.init(args=args)
    node = LaneFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()