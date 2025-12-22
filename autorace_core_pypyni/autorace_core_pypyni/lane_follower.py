import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class LaneFollower(Node):
    def __init__(self):
        super().__init__('lane_follower')
        
        qos_camera = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_command = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_lidar = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        # Подписки
        self.sub_camera = self.create_subscription(Image, '/color/image', self.camera_callback, qos_camera)
        self.sub_command = self.create_subscription(String, '/course_code', self.command_callback, qos_command)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_lidar)
        
        # Паблишеры
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # === ДОБАВЛЕНО: ПАБЛИШЕРЫ ДЛЯ ФИНИША ===
        self.pub_finish = self.create_publisher(String, '/robot_finish', 10)
        self.pub_course_code = self.create_publisher(String, '/course_code', 10)
        
        self.bridge = CvBridge()
        self.twist = Twist()
        self.stop_robot = True 

        self.scan_ranges = []
        self.lane_width_px = 300
        self.mode = 'center'
        
        # Таймер и флаг финиша
        self.no_cone_timer = 0
        self.is_finished = False
        self.team_name = "пупуни" # <--- ТВОЯ КОМАНДА

        # PID
        self.Kp = 0.005 
        self.Ki = 0.0000
        self.Kd = 0.005
        
        self.desiredV = 0.15 
        self.E = [0] * 15
        self.old_e = 0

        self.get_logger().info("Lane Follower: FULL SYSTEM READY")

    def scan_callback(self, msg):
        self.scan_ranges = [r if not math.isinf(r) else 10.0 for r in msg.ranges]

    def command_callback(self, msg):
        cmd = msg.data.lower().strip()
        if cmd in ['left', 'right', 'center', 'construction']:
            self.mode = cmd
            self.stop_robot = False
            self.no_cone_timer = 0
            self.get_logger().info(f"👉 MODE: {cmd.upper()}")
        elif cmd == 'stop':
            self.stop_robot = True
            self.get_logger().info("🛑 ROBOT STOPPED (Command)")
        elif cmd == 'go':
            self.stop_robot = False

    def camera_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: return

        height, width, _ = cv_image.shape
        crop_img = cv_image[int(height*0.7):height, 0:width]
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        
        # 1. Маски ЛИНИЙ
        lower_yellow = np.array([20, 80, 80]); upper_yellow = np.array([40, 255, 255])
        lower_white = np.array([0, 0, 180]); upper_white = np.array([180, 50, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # Чистим маски
        h_crop, w_crop = mask_yellow.shape
        mid = int(w_crop / 2)
        mask_yellow[:, mid:] = 0 
        mask_white[:, :mid] = 0 

        # 2. Маски КОНУСОВ
        mask_cone1 = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([10, 255, 255]))
        mask_cone2 = cv2.inRange(hsv, np.array([170, 100, 50]), np.array([180, 255, 255]))
        mask_cones = mask_cone1 | mask_cone2
        
        # 3. ВНЕДРЕНИЕ КОНУСОВ И ЛОГИКА ФИНИША
        if self.mode == 'construction':
            self.desiredV = 0.1
            
            # === ПРОВЕРКА НА ФИНИШ ===
            cone_pixels = cv2.countNonZero(mask_cones)
            if cone_pixels < 200: 
                self.no_cone_timer += 1
            else:
                self.no_cone_timer = 0
            
            # Если 60 кадров нет конусов и мы еще не финишировали
            if self.no_cone_timer > 140 and not self.is_finished:
                self.is_finished = True
                
                # 1. Отправляем имя команды
                self.pub_finish.publish(String(data=self.team_name))
                # 2. Отправляем команду СТОП (себе и другим)
                self.pub_course_code.publish(String(data="stop"))
                # 3. Физически останавливаемся
                self.stop_robot = True
                self.pub_cmd_vel.publish(Twist())
                
                self.get_logger().info(f"🏁 MISSION COMPLETE! Team: {self.team_name}")
                return

            # Рисование стен
            contours, _ = cv2.findContours(mask_cones, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1], reverse=True)
            
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area > 200: 
                    x, y, w_rect, h_rect = cv2.boundingRect(cnt)
                    cx_cone = x + w_rect // 2
                    pad = 25 
                    if cx_cone < width / 2:
                        cv2.rectangle(mask_yellow, (x-pad, y-pad), (x+w_rect+pad, y+h_rect+pad), 255, -1)
                        cv2.rectangle(crop_img, (x, y), (x+w_rect, y+h_rect), (0, 255, 255), 2)
                    else:
                        cv2.rectangle(mask_white, (x-pad, y-pad), (x+w_rect+pad, y+h_rect+pad), 255, -1)
                        cv2.rectangle(crop_img, (x, y), (x+w_rect, y+h_rect), (255, 255, 255), 2)
                    if i >= 2: break
        else:
            self.desiredV = 0.15
            self.no_cone_timer = 0 

        # 4. Поиск центров
        M_y = cv2.moments(mask_yellow)
        cy = int(M_y['m10']/M_y['m00']) if M_y['m00'] > 0 else None
        M_w = cv2.moments(mask_white)
        cw = int(M_w['m10']/M_w['m00']) if M_w['m00'] > 0 else None

        if cy and cw: self.lane_width_px = cw - cy

        target_center = width / 2 
        safety_offset = 40
        
        # === ЛОГИКА ===
        if self.mode == 'right':
            if cw: target_center = cw - (self.lane_width_px / 2) + safety_offset
            elif cy: target_center = cy + (self.lane_width_px / 2) + safety_offset
            else: target_center = width 
        
        elif self.mode == 'left':
            if cy: target_center = cy + (self.lane_width_px / 2) - safety_offset
            elif cw: target_center = cw - (self.lane_width_px / 2) - safety_offset
            else: target_center = 0 

        elif self.mode == 'construction':
            if cy and cw: target_center = (cy + cw) / 2
            elif cy: target_center = cy + (self.lane_width_px / 2)
            elif cw: target_center = cw - (self.lane_width_px / 2)
            
            # ЭКСТРЕННЫЙ ЛИДАР
            if len(self.scan_ranges) > 0:
                n = len(self.scan_ranges)
                sec_front = self.scan_ranges[int(n*(350/360)):] + self.scan_ranges[0:int(n*(10/360))]
                f_vals = [r for r in sec_front if r < 9.0]
                min_f = min(f_vals) if len(f_vals) > 0 else 10.0
                
                if min_f < 0.35: 
                    self.get_logger().info(f"🧱 WALL ({min_f:.2f})")
                    if cy and not cw: target_center = width
                    elif cw and not cy: target_center = 0
                    else: target_center = width

        else: # CENTER
            if cy and cw: target_center = (cy + cw) / 2
            elif cy: target_center = cy + (self.lane_width_px / 2)
            elif cw: target_center = cw - (self.lane_width_px / 2)

        error = (width / 2) - target_center

        # Debug HUD
        if cy: cv2.circle(crop_img, (cy, 20), 8, (0, 255, 255), -1)
        if cw: cv2.circle(crop_img, (cw, 20), 8, (255, 255, 255), -1)
        cv2.circle(crop_img, (int(target_center), 40), 6, (0, 255, 0), -1)
        
        if self.mode == 'construction':
            color = (0, 255, 0) if self.no_cone_timer < 180 else (0, 0, 255)
            cv2.putText(crop_img, f"Finish Timer: {self.no_cone_timer}/180", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Lane Tracking", crop_img)
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