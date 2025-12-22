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
        self.potential_sign = None
        self.sign_consistency_count = 0
        
        # === ЛОГИКА ЭТАПОВ ===
        # 0 = Ждем светофор
        # 1 = Ищем Синий знак
        # 2 = Ищем Красный знак
        # 3 = Стройка (Лидар)
        self.mission_stage = 0 
        
        self.decision_locked = False
        self.lock_timer = 0
        
        # === НОВОЕ: ЗАДЕРЖКА ПЕРЕД СТРОЙКОЙ ===
        self.construction_wait_active = False
        self.construction_wait_start = 0.0
        
        # Финиш
        self.team_name = "пупуни" 
        self.finish_timer_active = False
        self.finish_start_time = 0.0

        self.get_logger().info("👀 Perception: Ready (DELAYED CONSTRUCTION)")

    def img_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: return

        now = self.get_clock().now().nanoseconds / 1e9

        # 1. ТАЙМЕР ФИНИША
        if self.finish_timer_active:
            if (now - self.finish_start_time) >= 999.0: 
                self.finish_sequence()
                return

        # 2. ТАЙМЕР ОТЛОЖЕННОГО СТАРТА СТРОЙКИ (НОВОЕ)
        if self.construction_wait_active:
            elapsed = now - self.construction_wait_start
            # Ждем 6 секунд перед включением режима
            if elapsed >= 8.0:
                self.get_logger().info("🚧 TIMER DONE -> ACTIVATING CONSTRUCTION MODE")
                self.pub_command.publish(String(data="construction"))
                
                self.construction_wait_active = False
                self.mission_stage = 3 # Переходим в вечный режим стройки
                
                # Блокируем камеру ненадолго, чтобы не ловить глюки
                self.decision_locked = True
                self.lock_timer = 20 
            else:
                # Пока ждем - просто выходим, пусть едет по команде center
                return

        # 3. ЛОГИКА БЛОКИРОВКИ ПОСЛЕ ПОВОРОТА
        if self.decision_locked:
            self.lock_timer -= 1
            if self.lock_timer <= 0:
                self.decision_locked = False
                
                if self.mission_stage == 1:
                    # Закончили поворот -> едем ПРЯМО и ищем КРАСНЫЙ
                    self.pub_command.publish(String(data="center"))
                    self.mission_stage = 2 
                    self.get_logger().info("✅ TURN DONE -> SEARCHING RED")
            return

        # 4. ПОИСК ЗНАКОВ
        if self.mission_stage == 0:
            self.check_traffic_light(cv_image)
            
        elif self.mission_stage == 1:
            self.detect_sign(cv_image, target_color='blue')
            
        elif self.mission_stage == 2:
            self.detect_sign(cv_image, target_color='red')

    def check_traffic_light(self, img):
        h, w, _ = img.shape
        roi = img[0:int(h/2), 0:w] 
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 20, 20]), np.array([95, 255, 255]))
        
        if cv2.countNonZero(mask) > 50: 
            self.get_logger().info("🟢 GREEN LIGHT! GO!")
            self.start_moving = True
            self.pub_command.publish(String(data="go"))
            self.mission_stage = 1

    def detect_sign(self, img, target_color='blue'):
        h, w, _ = img.shape
        roi = img[0:int(h*0.8), :] 
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        kernel = np.ones((3,3), np.uint8)
        
        contours = []
        
        if target_color == 'blue':
            mask = cv2.inRange(hsv, np.array([80, 40, 30]), np.array([140, 255, 255]))
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
        elif target_color == 'red':
            mask1 = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([170, 100, 50]), np.array([180, 255, 255]))
            mask = mask1 | mask2
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        debug_frame = roi.copy()
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if target_color == 'blue' and area > 300:
                x, y, w_rect, h_rect = cv2.boundingRect(cnt)
                if x < 5 or (x + w_rect) > (w - 5): continue 
                
                ratio = float(w_rect) / h_rect
                if 0.5 < ratio < 2.0: 
                    cv2.rectangle(debug_frame, (x,y), (x+w_rect, y+h_rect), (0,255,0), 2)
                    sign_roi = roi[y:y+h_rect, x:x+w_rect]
                    direction = self.analyze_arrow_top_crop(sign_roi) 
                    if direction:
                        self.process_consistency(direction)
                        break 

            elif target_color == 'red' and area > 400:
                x, y, w_rect, h_rect = cv2.boundingRect(cnt)
                cv2.rectangle(debug_frame, (x,y), (x+w_rect, y+h_rect), (0,0,255), 2)
                cv2.putText(debug_frame, "CONSTRUCTION", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                self.process_consistency("construction")
                break

        cv2.imshow("Perception Debug", debug_frame)
        cv2.waitKey(1)

    def analyze_arrow_top_crop(self, sign_img):
        h, w, _ = sign_img.shape
        crop_h = int(h * 0.4) 
        crop = sign_img[0:crop_h, :]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 60]), np.array([180, 50, 255]))
        
        M = cv2.moments(mask)
        if M['m00'] < 10: return None 
        
        cx = int(M['m10'] / M['m00'])
        center_x = w / 2
        
        if cx < (center_x - 5): return 'left'
        elif cx > (center_x + 5): return 'right'
        return None

    def process_consistency(self, direction):
        if direction == self.potential_sign:
            self.sign_consistency_count += 1
        else:
            self.potential_sign = direction
            self.sign_consistency_count = 1
            
        if self.sign_consistency_count >= 2:
            self.get_logger().info(f"🔵 SIGN DETECTED: {direction.upper()}")
            
            if direction == "construction":
                # === ИЗМЕНЕНИЕ: НЕ ОТПРАВЛЯЕМ СРАЗУ ===
                self.get_logger().info("⏳ RED SIGN SEEN -> WAITING 6 SECONDS...")
                self.construction_wait_active = True
                self.construction_wait_start = self.get_clock().now().nanoseconds / 1e9
                
                # Блокируем детекцию знаков, чтобы не спамило
                self.decision_locked = True
                self.lock_timer = 9999 # Блокируем навсегда, пока таймер не сработает
                
            else:
                # ОБЫЧНЫЙ ПОВОРОТ
                self.pub_command.publish(String(data=direction))
                self.decision_locked = True
                self.lock_timer = 450 
            
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