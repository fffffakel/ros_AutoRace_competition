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
        
        # Настройка QoS (чтобы сообщения точно доходили, а видео не лагало)
        qos_camera = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_command = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)

        # Подписки
        self.sub_camera = self.create_subscription(Image, '/color/image', self.camera_callback, qos_camera)
        self.sub_command = self.create_subscription(String, '/course_code', self.command_callback, qos_command)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.bridge = CvBridge()
        self.twist = Twist()

        # PID Параметры
        self.Kp = 0.003
        self.Ki = 0.0001
        self.Kd = 0.005
        self.desiredV = 0.22
        
        self.E = [0] * 15
        self.old_e = 0
        
        # Переменные для хранения
        self.lane_width_px = 300 # Стандартная ширина (если не видим вторую линию)
        self.mode = 'center'     # Режимы: 'center', 'left', 'right'

        self.get_logger().info("Lane Follower: SIMPLE MODE READY")

    def command_callback(self, msg):
        # Получаем команду и убираем пробелы
        cmd = msg.data.lower().strip()
        
        if cmd in ['left', 'right', 'center']:
            self.mode = cmd
            self.get_logger().info(f"👉 MODE SWITCHED TO: {cmd.upper()}")
        else:
            self.get_logger().warn(f"Unknown command: {cmd}")

    def camera_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: return

        height, width, _ = cv_image.shape
        # Обрезаем верх, оставляем низ (дорогу)
        crop_img = cv_image[int(height*0.65):height, 0:width]
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        
        # === ЦВЕТА (ПРОВЕРЬ КАЛИБРОВКУ!) ===
        lower_yellow = np.array([20, 80, 80]); upper_yellow = np.array([40, 255, 255])
        lower_white = np.array([0, 0, 180]); upper_white = np.array([180, 50, 255])

        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # Чистим мусор (желтая слева, белая справа)
        h_crop, w_crop = mask_yellow.shape
        mid = int(w_crop / 2)
        mask_yellow[:, mid:] = 0 
        mask_white[:, :mid] = 0  

        # ==========================================
        # 🧠 ЛОГИКА "ПРИЖИМАНИЯ" К СТОРОНЕ
        # ==========================================
        
        target_center = width / 2 # По умолчанию едем прямо

        # Ищем центры линий
        M_y = cv2.moments(mask_yellow)
        cy = int(M_y['m10']/M_y['m00']) if M_y['m00'] > 0 else None

        M_w = cv2.moments(mask_white)
        cw = int(M_w['m10']/M_w['m00']) if M_w['m00'] > 0 else None

        # Обновляем ширину дороги, если видим обе (для точности)
        if cy is not None and cw is not None:
            self.lane_width_px = cw - cy

        # --- ВЫБОР ТРАЕКТОРИИ ---
        
        if self.mode == 'right':
            # Едем НАПРАВО -> Смотрим ТОЛЬКО на БЕЛУЮ линию
            if cw is not None:
                # Наша цель: держаться левее белой линии на половину ширины дороги
                target_center = cw - (self.lane_width_px / 2)
            else:
                # Если потеряли белую - паника, пытаемся найти (поворачиваем направо)
                target_center = width # Едем вправо искать линию
        
        elif self.mode == 'left':
            # Едем НАЛЕВО -> Смотрим ТОЛЬКО на ЖЕЛТУЮ линию
            if cy is not None:
                # Наша цель: держаться правее желтой линии на половину ширины дороги
                target_center = cy + (self.lane_width_px / 2)
            else:
                target_center = 0 # Едем влево искать линию

        else: 
            # Режим CENTER (обычная езда)
            if cy is not None and cw is not None:
                target_center = (cy + cw) / 2
            elif cy is not None:
                target_center = cy + (self.lane_width_px / 2)
            elif cw is not None:
                target_center = cw - (self.lane_width_px / 2)

        # Расчет ошибки
        error = (width / 2) - target_center
        
        # --- DEBUG VISUALIZATION ---
        debug = crop_img.copy()
        if cy: cv2.circle(debug, (cy, 20), 10, (0, 255, 255), -1)
        if cw: cv2.circle(debug, (cw, 20), 10, (255, 255, 255), -1)
        cv2.circle(debug, (int(target_center), 20), 5, (0, 255, 0), -1)
        
        cv2.putText(debug, f"MODE: {self.mode.upper()}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Robot Brain", debug)
        cv2.waitKey(1)
        
        self.pid_control(error)

    def pid_control(self, error):
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

if __name__ == '__main__':
    main()