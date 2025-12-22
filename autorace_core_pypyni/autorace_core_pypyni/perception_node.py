import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

class AutoRacePerception(Node):
    def __init__(self):
        super().__init__('perception_node')
        
        self.sub_camera = self.create_subscription(Image, '/color/image', self.img_callback, 10)
        self.pub_command = self.create_publisher(String, '/course_code', 10)
        self.pub_finish = self.create_publisher(String, '/robot_finish', 10)
        
        self.bridge = CvBridge()
        
        self.start_moving = False
        self.last_sign_cmd = "center"
        self.potential_sign = None
        self.sign_consistency_count = 0
        
        # Логика блокировки
        self.decision_locked = False
        self.lock_timer = 0
        
        # Финиш
        self.team_name = "пупуни" 
        self.finish_timer_active = False
        self.finish_start_time = 0.0

        self.get_logger().info("👀 Perception: AGGRESSIVE TOP CROP (40%)")

    def img_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: return

        if self.finish_timer_active:
            elapsed = (self.get_clock().now().nanoseconds / 1e9) - self.finish_start_time
            if elapsed >= 40.0:
                self.finish_sequence()
                return

        if self.decision_locked:
            self.lock_timer -= 1
            if self.lock_timer <= 0:
                self.decision_locked = False
            if not self.start_moving:
                self.check_traffic_light(cv_image)
            return

        if not self.start_moving:
            self.check_traffic_light(cv_image)
        else:
            self.detect_sign(cv_image)

    def check_traffic_light(self, img):
        h, w, _ = img.shape
        roi = img[0:int(h/2), 0:w] 
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 20, 20]), np.array([95, 255, 255]))
        
        if cv2.countNonZero(mask) > 50: 
            self.get_logger().info("🟢 GREEN LIGHT! GO!")
            self.start_moving = True
            self.pub_command.publish(String(data="go"))

    def detect_sign(self, img):
        h, w, _ = img.shape
        roi = img[0:int(h*0.8), :] 
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Синий
        mask_blue = cv2.inRange(hsv, np.array([80, 40, 30]), np.array([140, 255, 255]))
        kernel = np.ones((3,3), np.uint8)
        mask_blue = cv2.erode(mask_blue, kernel, iterations=1)
        mask_blue = cv2.dilate(mask_blue, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        debug_frame = roi.copy()
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:
                x, y, w_rect, h_rect = cv2.boundingRect(cnt)
                
                # Проверка границ
                if x < 5 or (x + w_rect) > (w - 5):
                    continue 

                ratio = float(w_rect) / h_rect
                if 0.5 < ratio < 2.0: 
                    # Рамка знака
                    cv2.rectangle(debug_frame, (x,y), (x+w_rect, y+h_rect), (0,255,0), 2)
                    
                    sign_roi = roi[y:y+h_rect, x:x+w_rect]
                    direction = self.analyze_arrow_top_crop(sign_roi) 
                    
                    if direction:
                        # Рисуем результат
                        cv2.putText(debug_frame, direction, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                        self.process_consistency(direction)
                        break 
        
        cv2.imshow("Perception Debug", debug_frame)
        cv2.waitKey(1)

    def analyze_arrow_top_crop(self, sign_img):
        """
        Берем только ВЕРХНИЕ 40% знака.
        Это отрезает вертикальную палку стрелки.
        Остается только горизонтальный "наконечник".
        """
        h, w, _ = sign_img.shape
        
        # === КЛЮЧЕВОЙ МОМЕНТ: РЕЖЕМ ЖЕСТКО ===
        crop_h = int(h * 0.4) 
        crop = sign_img[0:crop_h, :]
        
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Белый
        mask = cv2.inRange(hsv, np.array([0, 0, 60]), np.array([180, 50, 255]))
        
        # Для отладки показываем, что именно мы анализируем (обрезанную верхушку)
        cv2.imshow("Arrow Top Crop", mask)
        
        M = cv2.moments(mask)
        if M['m00'] < 10: return None 
        
        # Центр масс белого пятна
        cx = int(M['m10'] / M['m00'])
        center_x = w / 2
        
        # Если центр масс слева -> Left
        # Если центр масс справа -> Right
        
        # Добавляем небольшой порог (5px), чтобы не шумело по центру
        if cx < (center_x - 5):
            return 'left'
        elif cx > (center_x + 5):
            return 'right'
            
        return None

    def process_consistency(self, direction):
        if direction == self.potential_sign:
            self.sign_consistency_count += 1
        else:
            self.potential_sign = direction
            self.sign_consistency_count = 1
            
        # Ждем 2 кадра
        if self.sign_consistency_count >= 2:
            self.get_logger().info(f"🔵 SIGN LOCKED: {direction.upper()}")
            self.pub_command.publish(String(data=direction))
            
            # БЛОКИРУЕМ РЕШЕНИЕ
            self.decision_locked = True
            self.lock_timer = 300 
            self.get_logger().info("🔒 DECISION LOCKED")
            
            if not self.finish_timer_active:
                self.finish_timer_active = True
                self.finish_start_time = self.get_clock().now().nanoseconds / 1e9
                self.get_logger().info("⏳ FINISH TIMER: START")

            self.sign_consistency_count = 0

    def finish_sequence(self):
        msg = String(); msg.data = self.team_name
        self.pub_finish.publish(msg)
        self.pub_command.publish(String(data="stop"))
        self.get_logger().info(f"🏁 FINISH! Team: {self.team_name}")
        self.finish_timer_active = False
        self.start_moving = False

def main(args=None):
    rclpy.init(args=args)
    node = AutoRacePerception()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown(); cv2.destroyAllWindows()
if __name__ == '__main__': main()