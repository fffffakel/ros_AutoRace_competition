import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

# Узел perception_node.py
# Назначение: распознавание событий по камере и выдача режимных команд в /course_code.
#
# Структура работы:
#   - миссия разбита на этапы (mission_stage) и выполняется как конечный автомат (FSM).
#   - на каждом кадре камеры проверяется только то, что соответствует текущему этапу.
#   - решения защищены от дребезга:
#       * sign_consistency_count требует повторения распознавания,
#       * decision_locked на время блокирует новые решения после команды поворота,
#       * construction_wait_active откладывает включение режима construction на фиксированное время.
#
# Команды, которые публикуются:
#   go                 - разрешить движение (узел lane_follower начнёт выдавать /cmd_vel);
#   left/right/center  - режим следования линии;
#   construction       - режим коридора из конусов.
#
# Ввыбирает 'режим' и передаёт его исполнителю (lane_follower).


# Класс ROS2-ноды восприятия.
# Внутреннее состояние:
#   - start_moving: флаг старта после зелёного светофора;
#   - потенциальный знак и счётчик устойчивости;
#   - mission_stage: этап сценария;
#   - decision_locked/lock_timer: блокировка принятия решений после команды;
#   - construction_wait_*: задержка перед включением construction после красного знака;
#   - finish_*: таймер финиша (в текущем коде задан большой интервал, фактически не срабатывает быстро).


class AutoRacePerception(Node):
    def __init__(self):
        super().__init__("perception_node")

        # Интерфейсы ROS:
        #   /color/image   (Image)  -> вход камеры для детекции светофора и знаков.
        #   /course_code   (String) -> выход команд режима движения.
        #   /robot_finish  (String) -> оповещение о финише (имя команды).

        self.sub_camera = self.create_subscription(
            Image, "/color/image", self.img_callback, 10
        )
        self.pub_command = self.create_publisher(String, "/course_code", 10)
        self.pub_finish = self.create_publisher(String, "/robot_finish", 10)

        self.bridge = CvBridge()

        # Переменные устойчивости распознавания:
        #   potential_sign          - последнее распознанное направление ('left'/'right'/'construction').
        #   sign_consistency_count  - сколько кадров подряд подтверждают тот же результат.
        # Цель: не реагировать на единичный шумовой контур.

        self.start_moving = False
        self.potential_sign = None
        self.sign_consistency_count = 0

        # FSM (конечный автомат) миссий:
        #   0: ожидание зелёного светофора (разрешение на старт).
        #   1: поиск синего знака-направления (стрелка left/right).
        #   2: поиск красного знака (переход к строительной зоне).
        #   3: режим construction активирован (дальше этот узел уже не ищет знаки, т.к. управление в коридоре выполняет lane_follower).

        self.mission_stage = 0

        # Блокировка решений:
        # После того как мы отдали команду поворота, несколько сотен кадров игнорируем новые знаки.
        # Это предотвращает повторное срабатывание на тот же знак, пока робот ещё выполняет манёвр.

        self.decision_locked = False
        self.lock_timer = 0

        # Задержка перед включением construction:
        # После красного знака робот должен ещё некоторое время ехать прямо/по центру,
        # чтобы доехать до зоны конусов и не включить construction слишком рано.
        # Для этого включаем construction_wait_active и ждём заданное число секунд.

        self.construction_wait_active = False
        self.construction_wait_start = 0.0

        # Финишная последовательность:
        # Публикация /robot_finish (имя команды) и 'stop' в /course_code.
        # В данном коде финиш управляется таймером finish_timer_active.
        # Примечание: реальное завершение миссии часто определяется рефери/правилами; здесь заложен механизм уведомления.

        self.team_name = "пупуни"
        self.finish_timer_active = False
        self.finish_start_time = 0.0

        self.get_logger().info("👀 Perception: Ready (DELAYED CONSTRUCTION)")

    # img_callback(msg: Image)
    # Обработка каждого кадра:
    #   1) Конвертируем изображение.
    #   2) Проверяем таймер финиша (если активирован).
    #   3) Если активна задержка construction_wait_active — ждём заданное время, затем публикуем 'construction'.
    #   4) Если decision_locked — уменьшаем lock_timer и при его окончании переводим FSM на следующий этап.
    #   5) В зависимости от mission_stage выполняем: check_traffic_light / detect_sign(blue/red).

    def img_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            return

        # now — текущее ROS-время в секундах (float).
        # Используется для таймеров ожидания и (при необходимости) финишного таймера.

        now = self.get_clock().now().nanoseconds / 1e9

        # Финишный таймер:
        # Если активирован, по истечении заданного интервала вызывается finish_sequence().
        # В текущей конфигурации интервал очень большой, поэтому обычно этот путь не используется в ходе короткого заезда.

        if self.finish_timer_active:
            if (now - self.finish_start_time) >= 999.0:
                self.finish_sequence()
                return

        # Отложенное включение construction:
        # Пока таймер не закончился — выходим из callback, не анализируя знаки.
        # Это важно: на подъезде к стройке могут быть ложные красные/синие элементы.

        if self.construction_wait_active:
            elapsed = now - self.construction_wait_start

            if elapsed >= 8.0:
                self.get_logger().info("🚧 TIMER DONE -> ACTIVATING CONSTRUCTION MODE")
                self.pub_command.publish(String(data="construction"))

                self.construction_wait_active = False
                self.mission_stage = 3

                self.decision_locked = True
                self.lock_timer = 20
            else:

                return

        # decision_locked режим:
        # Каждый кадр уменьшаем lock_timer.
        # Когда таймер закончился:
        #   - снимаем блокировку;
        #   - если это был поворот по синему знаку (stage 1), переводим FSM на поиск красного знака (stage 2) и отдаём 'center'.

        if self.decision_locked:
            self.lock_timer -= 1
            if self.lock_timer <= 0:
                self.decision_locked = False

                if self.mission_stage == 1:

                    self.pub_command.publish(String(data="center"))
                    self.mission_stage = 2
                    self.get_logger().info("✅ TURN DONE -> SEARCHING RED")
            return

        # Основной диспетчер FSM:
        # На каждом кадре выполняется только соответствующий детектор.
        # Это экономит вычисления и уменьшает вероятность конфликтующих решений.

        if self.mission_stage == 0:
            self.check_traffic_light(cv_image)

        elif self.mission_stage == 1:
            self.detect_sign(cv_image, target_color="blue")

        elif self.mission_stage == 2:
            self.detect_sign(cv_image, target_color="red")

    # check_traffic_light(img)
    # Ищет зелёный цвет светофора в верхней половине кадра:
    #   - выделяем ROI (верх кадра);
    #   - HSV-маска зелёного;
    #   - если пикселей достаточно, публикуем 'go' и переводим FSM на этап поиска синего знака.

    def check_traffic_light(self, img):
        h, w, _ = img.shape
        roi = img[0 : int(h / 2), 0:w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 20, 20]), np.array([95, 255, 255]))

        if cv2.countNonZero(mask) > 50:
            self.get_logger().info("🟢 GREEN LIGHT! GO!")
            self.start_moving = True
            self.pub_command.publish(String(data="go"))
            self.mission_stage = 1

    # detect_sign(img, target_color)
    # Унифицированный детектор знаков:
    #   - берём ROI (верхние 80% кадра), чтобы захватывать знаки над дорогой;
    #   - строим HSV-маску по цвету:
    #       blue -> поиск синего фона знака направления;
    #       red  -> поиск красного знака 'стройка';
    #   - лёгкая морфология (erode/dilate) для удаления шума;
    #   - поиск контуров и фильтрация по площади/форме.
    #
    # Результат детекции не принимается мгновенно: далее вызывается process_consistency(),
    # которая требует повторения результата несколько кадров.

    def detect_sign(self, img, target_color="blue"):
        h, w, _ = img.shape
        roi = img[0 : int(h * 0.8), :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        kernel = np.ones((3, 3), np.uint8)

        contours = []

        if target_color == "blue":
            mask = cv2.inRange(hsv, np.array([80, 40, 30]), np.array([140, 255, 255]))
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        elif target_color == "red":
            mask1 = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(
                hsv, np.array([170, 100, 50]), np.array([180, 255, 255])
            )
            mask = mask1 | mask2
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        debug_frame = roi.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if target_color == "blue" and area > 300:
                x, y, w_rect, h_rect = cv2.boundingRect(cnt)
                if x < 5 or (x + w_rect) > (w - 5):
                    continue

                ratio = float(w_rect) / h_rect
                if 0.5 < ratio < 2.0:
                    cv2.rectangle(
                        debug_frame, (x, y), (x + w_rect, y + h_rect), (0, 255, 0), 2
                    )
                    sign_roi = roi[y : y + h_rect, x : x + w_rect]
                    direction = self.analyze_arrow_top_crop(sign_roi)
                    if direction:
                        self.process_consistency(direction)
                        break

            elif target_color == "red" and area > 400:
                x, y, w_rect, h_rect = cv2.boundingRect(cnt)
                cv2.rectangle(
                    debug_frame, (x, y), (x + w_rect, y + h_rect), (0, 0, 255), 2
                )
                cv2.putText(
                    debug_frame,
                    "CONSTRUCTION",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                self.process_consistency("construction")
                break

        cv2.imshow("Perception Debug", debug_frame)
        cv2.waitKey(1)

    # analyze_arrow_top_crop(sign_img)
    # Грубая оценка направления стрелки:
    #   - берём верхнюю часть знака (crop_h ~ 40%), где обычно расположена стрелка;
    #   - выделяем светлые/белые элементы (стрелка) маской низкой насыщенности;
    #   - считаем центр массы белых пикселей (cx) и сравниваем с серединой изображения знака.
    # Если cx смещён влево/вправо — возвращаем 'left'/'right'.

    def analyze_arrow_top_crop(self, sign_img):
        h, w, _ = sign_img.shape
        crop_h = int(h * 0.4)
        crop = sign_img[0:crop_h, :]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 60]), np.array([180, 50, 255]))

        M = cv2.moments(mask)
        if M["m00"] < 10:
            return None

        cx = int(M["m10"] / M["m00"])
        center_x = w / 2

        if cx < (center_x - 5):
            return "left"
        elif cx > (center_x + 5):
            return "right"
        return None

    # process_consistency(direction)
    # Антидребезг:
    #   - если направление совпадает с potential_sign, увеличиваем sign_consistency_count;
    #   - иначе считаем, что появилась новая гипотеза, и начинаем счёт заново.
    #
    # Когда счётчик достигает порога:
    #   - для left/right публикуем соответствующую команду и включаем decision_locked на время выполнения поворота;
    #   - для construction активируем construction_wait_active и блокируем обработку знаков до окончания таймера.
    #
    # Такой подход уменьшает вероятность ложного поворота/ранней стройки из-за единичного шума.

    def process_consistency(self, direction):
        if direction == self.potential_sign:
            self.sign_consistency_count += 1
        else:
            self.potential_sign = direction
            self.sign_consistency_count = 1

        if self.sign_consistency_count >= 2:
            self.get_logger().info(f"🔵 SIGN DETECTED: {direction.upper()}")

            if direction == "construction":

                self.get_logger().info("⏳ RED SIGN SEEN -> WAITING 6 SECONDS...")
                self.construction_wait_active = True
                self.construction_wait_start = self.get_clock().now().nanoseconds / 1e9

                self.decision_locked = True
                self.lock_timer = 9999

            else:

                self.pub_command.publish(String(data=direction))
                self.decision_locked = True
                self.lock_timer = 450

            self.sign_consistency_count = 0

    # finish_sequence()
    # Публикует сообщение о финише и останавливает движение:
    #   - /robot_finish <- team_name
    #   - /course_code  <- 'stop'
    # После этого сбрасывает флаги таймера.

    def finish_sequence(self):
        msg = String()
        msg.data = self.team_name
        self.pub_finish.publish(msg)
        self.pub_command.publish(String(data="stop"))
        self.get_logger().info(f"🏁 FINISH! Team: {self.team_name}")
        self.finish_timer_active = False
        self.start_moving = False


# Точка входа ROS2.
# Запускает ноду восприятия и корректно закрывает OpenCV-окна при завершении.


def main(args=None):
    rclpy.init(args=args)
    node = AutoRacePerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
