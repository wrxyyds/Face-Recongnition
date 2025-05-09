import logging
import os
import threading
import numpy as np
import train
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QEvent, Signal, QTimer
from PySide6.QtWidgets import *
from PySide6.QtGui import QImage, QPixmap
from qt_material import apply_stylesheet, QtStyleTools, QUiLoader
from face import *
from PIL import Image
import pymysql
from datetime import datetime




class QShowImage(QWidget):
    def __init__(self):
        super(QShowImage, self).__init__()
        self.label = QLabel(self)
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        # 设置窗口标题
        self.setWindowTitle("Image")

    def set_image(self, q_image):
        pixmap = QPixmap.fromImage(q_image)
        self.label.setPixmap(pixmap)


class RegistrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("注册信息")

        # 创建输入框
        self.student_id_input = QLineEdit(self)
        self.name_input = QLineEdit(self)
        self.identity_input = QLineEdit(self)

        # 创建标签
        student_id_label = QLabel("学号:", self)
        name_label = QLabel("姓名:", self)
        identity_label = QLabel("身份:", self)

        # 创建按钮
        ok_button = QPushButton("确定", self)
        cancel_button = QPushButton("取消", self)

        # 布局
        layout = QGridLayout()
        layout.addWidget(student_id_label, 0, 0)
        layout.addWidget(self.student_id_input, 0, 1)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(self.name_input, 1, 1)
        layout.addWidget(identity_label, 2, 0)
        layout.addWidget(self.identity_input, 2, 1)
        layout.addWidget(ok_button, 3, 0)
        layout.addWidget(cancel_button, 3, 1)

        self.setLayout(layout)

        # 连接按钮信号
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)

    def get_inputs(self):
        student_id = self.student_id_input.text()
        name = self.name_input.text()
        identity = self.identity_input.text()
        return student_id, name, identity


class FaceCaptureDialog(QDialog):
    """Dialog for capturing multiple face images with visual feedback"""
    capture_complete = Signal(list)  # Signal to emit when capture is complete with list of face images

    def __init__(self, parent=None, face_system=None):
        super().__init__(parent)
        self.setWindowTitle("Face Capture")
        self.setMinimumSize(800, 600)

        self.face_system = face_system
        self.cap = None  # Initialize camera later
        self.faces = []
        self.face_images = []
        self.capture_count = 0
        self.required_captures = 5  # Capture 5 images for better training
        self.is_capturing = True
        self.current_frame = None  # Store current frame

        # Create UI components
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.status_label = QLabel("Initializing camera...")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.capture_button = QPushButton("捕捉图片")
        self.capture_button.clicked.connect(self.capture_face)
        self.capture_button.setEnabled(False)  # Disabled until camera is ready

        self.complete_button = QPushButton("完成")
        self.complete_button.setEnabled(False)
        self.complete_button.clicked.connect(self.accept)

        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)

        # Face preview area
        self.preview_layout = QHBoxLayout()
        self.preview_labels = []
        for i in range(self.required_captures):
            label = QLabel()
            label.setFixedSize(100, 100)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border: 1px solid gray;")
            self.preview_labels.append(label)
            self.preview_layout.addWidget(label)

        # Create button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.capture_button)
        button_layout.addWidget(self.complete_button)
        button_layout.addWidget(self.cancel_button)

        # Create main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.camera_label)
        main_layout.addWidget(self.status_label)
        main_layout.addLayout(self.preview_layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

        # Initialize camera with a delay to give UI time to show
        QTimer.singleShot(300, self.initialize_camera)

        # Start the camera update timer (will be activated after camera initialization)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera)

    def initialize_camera(self):
        """Initialize the camera with proper error handling"""
        try:
            self.status_label.setText("Initializing camera...")
            QApplication.processEvents()

            # Try to open the camera
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow on Windows

            # Set camera properties for better stability
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Check if camera is opened successfully
            if not self.cap.isOpened():
                self.status_label.setText("Failed to open camera. Please check connections.")
                logger.error("Failed to open camera.")
                return

            # Warm up the camera by reading a few frames
            for _ in range(5):
                self.cap.read()
                time.sleep(0.1)

            self.status_label.setText(f"Camera ready! (0/{self.required_captures} images)")
            self.timer.start(33)  # ~30 fps
            logger.info("Camera initialized successfully in FaceCaptureDialog")

        except Exception as e:
            self.status_label.setText(f"Camera error: {str(e)}")
            logger.error(f"Camera initialization error: {e}")

    def update_camera(self):
        """Update the camera feed and detect faces"""
        if not self.cap or not self.cap.isOpened():
            return

        try:
            # Read a frame with proper error handling
            ret, frame = self.cap.read()
            if not ret:
                self.status_label.setText("Camera read error. Retrying...")
                logger.warning("Camera read error")
                # Try to recover
                self.timer.stop()
                QTimer.singleShot(500, self.recover_camera)
                return

            # Store current frame
            self.current_frame = frame.copy()

            # Process frame for display
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL for face detection
            pil_image = Image.fromarray(frame)
            pil_image_resized = pil_image.resize((512, 512))

            # Detect faces
            faces = self.face_system.face_detect(pil_image_resized)
            faces = np.array(faces)

            # Draw face boxes on the frame
            if faces.shape[0] > 0:
                # Convert face coordinates back to the frame's scale
                scaled_faces = faces.copy()
                scaled_faces[:, 0] = scaled_faces[:, 0] / 512 * frame.shape[1]
                scaled_faces[:, 1] = scaled_faces[:, 1] / 512 * frame.shape[0]
                scaled_faces[:, 2] = scaled_faces[:, 2] / 512 * frame.shape[1]
                scaled_faces[:, 3] = scaled_faces[:, 3] / 512 * frame.shape[0]
                scaled_faces = scaled_faces.astype(int)

                # Draw rectangles
                for face in scaled_faces:
                    cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2)

                # Store detected faces for capture
                self.faces = faces

                # Update status and enable capture button
                if self.capture_count == 0:
                    self.status_label.setText(
                        f"Face detected! Press 'Capture Face' ({self.capture_count}/{self.required_captures} images)")
                    self.capture_button.setEnabled(True)
                else:
                    self.status_label.setText(
                        f"Face detected! Press 'Capture Face' ({self.capture_count}/{self.required_captures} images)")
            else:
                self.faces = []
                self.status_label.setText("No face detected. Please position your face in the camera.")
                self.capture_button.setEnabled(False)

            # Display the frame
            h, w, ch = frame.shape
            img = QImage(frame.data, w, h, w * ch, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(img).scaled(
                self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        except Exception as e:
            logger.error(f"Error in update_camera: {e}")
            self.status_label.setText(f"Camera error. Retrying...")
            # Try to recover
            self.timer.stop()
            QTimer.singleShot(500, self.recover_camera)

    def recover_camera(self):
        """Attempt to recover camera if it fails"""
        try:
            logger.info("Attempting camera recovery in FaceCaptureDialog")

            # Properly release the camera if it exists
            if self.cap and self.cap.isOpened():
                self.cap.release()
                logger.info("Released camera for recovery")

            # Wait a moment before reopening
            time.sleep(0.5)

            # Reopen camera
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if self.cap.isOpened():
                logger.info("Camera recovered successfully")
                self.status_label.setText(f"Camera ready! ({self.capture_count}/{self.required_captures} images)")
                self.timer.start(33)
            else:
                logger.error("Failed to recover camera")
                self.status_label.setText("Failed to recover camera. Please restart.")
        except Exception as e:
            logger.error(f"Error recovering camera: {e}")
            self.status_label.setText("Camera recovery failed.")

    def capture_face(self):
        """Capture the current face in the frame"""
        if len(self.faces) == 0 or self.current_frame is None:
            self.status_label.setText("No face detected or no frame available.")
            return

        try:
            # Log what we're working with
            logger.info(f"Capturing face. Face array shape: {np.array(self.faces).shape}")
            logger.info(f"Current frame shape: {self.current_frame.shape}")

            # Use the stored frame to avoid reading a new one
            frame = self.current_frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            pil_image_resized = pil_image.resize((512, 512))

            # Log detection attempt
            logger.info("Detecting faces in resized image")

            # Get the largest face in the image - expand this section with more logging
            faces = self.face_system.face_detect(pil_image_resized)
            faces = np.array(faces)

            logger.info(f"Detected faces shape: {faces.shape}")
            if faces.shape[0] > 0:
                logger.info(f"First face coordinates: {faces[0]}")

            if faces.shape[0] > 0:
                # Calculate areas to find largest face
                areas = (faces[:, 2] - faces[:, 0]) * (faces[:, 3] - faces[:, 1])
                largest_idx = np.argmax(areas)
                face = faces[largest_idx]

                logger.info(
                    f"Selected largest face: {face}, type: {type(face)}, shape: {face.shape if hasattr(face, 'shape') else 'N/A'}")

                # Handle face detection results carefully - this was the source of the error
                try:
                    # Make sure we're working with a properly formatted face detection
                    if not isinstance(face, np.ndarray) or len(face) < 4:
                        self.status_label.setText("Invalid face detection result. Try again.")
                        logger.error(f"Invalid face data: {face}")
                        return

                    # Extract the first 4 values safely
                    x1, y1, x2, y2 = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    logger.info(f"Extracted coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

                    # Convert coordinates to the original frame's scale
                    orig_x1 = int(x1 * frame.shape[1] / 512)
                    orig_y1 = int(y1 * frame.shape[0] / 512)
                    orig_x2 = int(x2 * frame.shape[1] / 512)
                    orig_y2 = int(y2 * frame.shape[0] / 512)

                    logger.info(f"Scaled to original: x1={orig_x1}, y1={orig_y1}, x2={orig_x2}, y2={orig_y2}")

                    # Add some padding (10% on each side)
                    pad_x = int((orig_x2 - orig_x1) * 0.1)
                    pad_y = int((orig_y2 - orig_y1) * 0.1)

                    # Apply padding with boundary checks
                    final_x1 = max(0, orig_x1 - pad_x)
                    final_y1 = max(0, orig_y1 - pad_y)
                    final_x2 = min(frame.shape[1], orig_x2 + pad_x)
                    final_y2 = min(frame.shape[0], orig_y2 + pad_y)

                    logger.info(
                        f"Final coordinates with padding: x1={final_x1}, y1={final_y1}, x2={final_x2}, y2={final_y2}")
                    logger.info(f"Frame boundaries: width={frame.shape[1]}, height={frame.shape[0]}")

                    # Check if coordinates make sense
                    if final_x1 >= final_x2 or final_y1 >= final_y2:
                        self.status_label.setText("Invalid face region. Try again.")
                        logger.error(f"Invalid face region: ({final_x1}, {final_y1}, {final_x2}, {final_y2})")
                        return

                    # Extract face image
                    face_img = np.ascontiguousarray(frame[final_y1:final_y2, final_x1:final_x2])
                    logger.info(f"Extracted face image shape: {face_img.shape}")

                    # Store face image
                    self.face_images.append(face_img)

                    # Display preview
                    h, w, _ = face_img.shape
                    q_img = QImage(face_img.data, w, h, w * 3, QImage.Format_RGB888)
                    self.preview_labels[self.capture_count].setPixmap(
                        QPixmap.fromImage(q_img).scaled(
                            100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))

                    # Increment counter
                    self.capture_count += 1

                    # Update status
                    self.status_label.setText(
                        f"Image {self.capture_count} captured! ({self.capture_count}/{self.required_captures})")

                    # Check if we have enough images
                    if self.capture_count >= self.required_captures:
                        self.status_label.setText("All images captured! Click 'Complete' to continue.")
                        self.capture_button.setEnabled(False)
                        self.complete_button.setEnabled(True)

                    # Add small delay to avoid duplicate captures
                    QApplication.processEvents()
                    time.sleep(0.5)

                except Exception as e:
                    logger.error(f"Error processing face coordinates: {e}")
                    self.status_label.setText(f"Error processing face: {str(e)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")

            else:
                self.status_label.setText("No faces detected. Please try again.")

        except Exception as e:
            logger.error(f"Error in capture_face: {e}")
            self.status_label.setText(f"Error capturing face: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def closeEvent(self, event):
        """Handle dialog close event"""
        logger.info("FaceCaptureDialog closing")
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("Camera released in FaceCaptureDialog closeEvent")
        event.accept()

    def accept(self):
        """Handle dialog acceptance"""
        logger.info("FaceCaptureDialog accepted")
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("Camera released in FaceCaptureDialog accept")
        if self.face_images:
            self.capture_complete.emit(self.face_images)
        super().accept()

    def reject(self):
        """Handle dialog rejection"""
        logger.info("FaceCaptureDialog rejected")
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("Camera released in FaceCaptureDialog reject")
        super().reject()

class CustomLoader(QUiLoader):
    def __init__(self, base_instance=None):
        super().__init__()
        self.base_instance = base_instance

    def createWidget(self, class_name, parent=None, name=''):
        if parent is None and self.base_instance:
            return self.base_instance
        return super().createWidget(class_name, parent, name)


class FaceWindow(QMainWindow, QtStyleTools):
    shutdown_signal = Signal()  # Define a signal for safe shutdown

    def __init__(self, parent=None):
        super().__init__(parent)
        self.open_flag = False
        self.frame = None
        self.another_frame = None
        self.image = None
        self.faces = None
        self.another_faces = None
        self.features = None
        self.register_ids = None
        self.names = None
        self.identities = None
        self.lock = threading.Lock()  # 线程锁
        self.is_shutting_down = False  # 关闭状态标志

        # 使用CustomLoader加载UI
        loader = CustomLoader(self)
        loader.load('ui.ui')

        # 连接信号
        self.button1.clicked.connect(self.video_open_close)
        self.button2.clicked.connect(self.improved_register)
        self.button3.clicked.connect(self.face_recognition)

        # 应用样式表
        apply_stylesheet(self, theme='light_blue.xml')

        # 连接关闭信号到清理方法
        self.shutdown_signal.connect(self.cleanup_resources)

        # 连接应用退出信号
        QApplication.instance().aboutToQuit.connect(self.on_app_quit)

        # 初始化摄像头（但不立即打开）
        self.cap = None
        self.fs = FaceSystem()
        self.face_img = QShowImage()

        # 显示窗口
        self.show()

        # 启动显示线程
        self.th = threading.Thread(target=self.display)
        self.th.daemon = True  # 设为守护线程以便主线程退出时自动结束
        self.th.start()
        self.load_feature()

        # 建立数据库连接
        try:
            self.connection = pymysql.connect(
                host='localhost',
                user='root',
                password='88888888',
                database='face_reconginition',
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            self.cursor = self.connection.cursor()
            logger.info("Database connection established")
        except pymysql.Error as e:
            logger.error(f"Database connection error: {e}")
            self.connection = None
            self.cursor = None
            QtWidgets.QMessageBox.critical(self, '错误', f'数据库连接失败: {e}')

    def save_feature(self, register_id, name, identity):
        # save np feature to file
        # id = len(os.listdir('../datas/faces'))
        # np.save(f'../datas/faces/{id}.npy', feature)
        with open('../datas/names.txt', 'a', encoding='utf-8') as f:
            f.write(str(register_id))
            f.write(',')
            f.write(name)
            f.write(',')
            f.write(identity)
            f.write('\n')
        '''if self.features.shape[0] != 0:
            self.features = np.concatenate((self.features, feature), axis=0)
        else:
            self.features = feature.reshape(1, -1)
        self.register_ids.append(register_id)
        self.names.append(name)
        self.identities.append(identity)'''

    def load_feature(self):
        self.features = []
        self.register_ids = []
        self.names = []
        self.identities = []
        feature_files = os.listdir('../datas/faces')
        if len(feature_files) > 0:
            for file in feature_files:
                self.features.append(np.load(f'../datas/faces/{file}'))
            with open('../datas/names.txt', 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    register_id, name, identity = line.split(',')
                    self.register_ids.append(register_id)
                    self.names.append(name)
                    self.identities.append(identity)
            self.features = np.array(self.features)
            self.features = np.squeeze(self.features)
        else:
            self.features = np.empty((0, 0))

    def calculate_distance(self, feature):
        # self.features的形状是(num, batch_size = 1, 512)
        distance = np.sqrt(np.sum(np.square(self.features - feature), axis=1))
        min_index = np.argmin(distance)
        return min_index, distance[min_index]

    def face_recognition(self):
        with self.lock:  # 使用线程锁
            if self.faces is None or self.faces.shape[0] == 0:
                QtWidgets.QMessageBox.warning(self, '提示', f'未识别到人脸')
                return

        face = np.ascontiguousarray(
            self.frame[self.faces[0][1]:self.faces[0][3], self.faces[0][0]:self.faces[0][2]])
        feature = self.fs.get_face_feature(Image.fromarray(face))
        min_index, min_dis = self.calculate_distance(feature)
        print('min_dis', min_dis)

        if min_dis < 0.8:
            now = datetime.now()
            punch_time = now.strftime('%Y-%m-%d %H:%M:%S')

            if not self.connection:
                QtWidgets.QMessageBox.warning(self, '错误', '数据库连接不可用')
                return

            cursor = None
            try:
                # 插入数据的 SQL 语句
                cursor = self.connection.cursor()
                if self.identities[min_index] == 'student':
                    sql = "INSERT INTO student_attendance (student_id, student_name, punch_time) VALUES (%s, %s, %s)"
                    values = (str(self.register_ids[min_index]), str(self.names[min_index]), punch_time)
                else:
                    sql = "INSERT INTO teacher_attendance (teacher_id, teacher_name, punch_time) VALUES (%s, %s, %s)"
                    values = (str(self.register_ids[min_index]), str(self.names[min_index]), punch_time)

                # 执行插入操作
                cursor.execute(sql, values)

                # 提交事务
                self.connection.commit()
                print("数据插入成功！")

                QtWidgets.QMessageBox.information(self, '提示', f'人脸识别结果为{self.register_ids[min_index]}')
            except Exception as e:
                # 发生错误时回滚
                if self.connection:
                    self.connection.rollback()
                print(f"数据库错误: {e}")
                QtWidgets.QMessageBox.warning(self, '错误', f'数据库操作失败: {e}')
            finally:
                # 确保手动关闭游标
                if cursor:
                    cursor.close()
        else:
            QtWidgets.QMessageBox.information(self, '提示', '人脸不在数据库中')

    def improved_register(self):
        """Improved registration process with multi-face capture"""
        # Check if camera is open
        if not self.open_flag:
            QMessageBox.warning(self, '提示', '请先打开摄像头')
            return

        # 暂时停止主显示循环访问摄像头
        temp_open_flag = self.open_flag
        self.open_flag = False  # Temporarily stop the display thread from accessing the camera
        time.sleep(0.2)  # Give display thread time to stop using camera

        # 关闭摄像头资源，确保完全释放
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            self.cap.release()
            print("Main camera released for registration")
            time.sleep(0.5)  # Wait for camera to fully release

        try:
            # Create and show face capture dialog
            capture_dialog = FaceCaptureDialog(self, self.fs)

            def process_captured_faces(face_images):
                """Process the captured face images"""
                if not face_images:
                    return

                # Get first face for feature extraction (can be modified to use average of all faces)
                face = face_images[0]

                # Extract feature
                feature = self.fs.get_face_feature(Image.fromarray(face))

                # Check if face already exists in the database
                if self.features.shape[0] != 0:
                    min_index, min_dis = self.calculate_distance(feature)
                    if min_dis < 0.8:
                        QMessageBox.warning(self, '提示', f'该人脸已经注册过!')
                        return

                # Get registration info
                dialog = RegistrationDialog(self)
                if dialog.exec() == QDialog.Accepted:
                    register_id, name, identity = dialog.get_inputs()
                    if not register_id or not name or not identity:
                        QMessageBox.warning(self, '提示', '请填写完整信息')
                        return

                    # Save all captured face images for better training
                    os.makedirs(f'../images/train/{register_id}', exist_ok=True)
                    for i, face_img in enumerate(face_images):
                        self.save_image(face_img, f'../images/train/{register_id}/face_{i}.jpg')

                    # Train the model with new images
                    train.train()

                    # Reload face system to include new model
                    self.fs = FaceSystem()

                    # Extract feature with updated model for better accuracy
                    feature = self.fs.get_face_feature(Image.fromarray(face_images[0]))

                    # Save feature to database
                    self.save_feature(str(register_id), str(name), str(identity))
                    self.reload()
                    print('reload finish')

                    QMessageBox.information(self, '提示', f'人脸注册成功！')

            # Connect signal to process captured faces
            capture_dialog.capture_complete.connect(process_captured_faces)

            # Show dialog
            capture_dialog.exec()

        except Exception as e:
            logger.error(f"Error in registration process: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            QMessageBox.critical(self, '错误', f'注册过程发生错误: {str(e)}')

        finally:
            # 确保这里重新打开摄像头，无论前面的操作是否成功
            # 重要：重新分配一个新的摄像头对象，而不是尝试重用之前可能处于不可靠状态的对象
            print("Reopening main camera after registration...")
            self.cap = None  # 清除旧对象，确保垃圾收集

            try:
                # 使用DirectShow后端重新打开摄像头（Windows系统）
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

                if self.cap is not None:  # Check if cap was created successfully
                    # 设置摄像头属性
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)

                    # 确认摄像头已成功打开
                    if self.cap.isOpened():
                        print("Main camera successfully reopened")
                        # 等待确保摄像头稳定
                        for _ in range(3):
                            self.cap.read()  # 读取几帧以稳定摄像头
                            time.sleep(0.1)

                        # 仅当之前摄像头是打开状态时才恢复显示
                        if temp_open_flag:
                            self.open_flag = True
                            print("Display thread resumed")
                    else:
                        print("Failed to reopen main camera")
                        logger.error("Failed to reopen main camera")
                        QMessageBox.warning(self, '警告', '无法重新打开摄像头，请尝试手动打开')
                else:
                    print("Failed to create camera object")
                    logger.error("Failed to create camera object")
                    QMessageBox.warning(self, '警告', '摄像头对象创建失败，请尝试手动打开')
            except Exception as e:
                print(f"Error reopening camera: {e}")
                logger.error(f"Error reopening camera: {e}")
                QMessageBox.warning(self, '警告', f'重新打开摄像头失败: {str(e)}')
                # 如果重新打开失败，确保设置正确的状态
                self.open_flag = False
                self.button1.setText('打开摄像头')

    def reload(self):
        try:
            # 1. 读取注册信息
            names_file = '../datas/names.txt'
            if not os.path.exists(names_file):
                QMessageBox.warning(self, '警告', '注册信息文件不存在')
                return

            with open(names_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if not lines:
                # QMessageBox.warning(self, '警告', '没有注册用户信息')
                return

            # 2. 遍历每个注册用户，更新特征
            for i, line in enumerate(lines):
                register_id, name, identity = line.strip().split(',')

                # 查找对应图片文件夹
                image_folder = f'../images/train/{register_id}'
                if not os.path.exists(image_folder):
                    logger.warning(f"用户{register_id}的图片文件夹不存在")
                    continue

                # 获取文件夹中的第一张图片
                image_files = [f for f in os.listdir(image_folder)
                               if f.endswith(('.jpg', '.jpeg', '.png'))]
                if not image_files:
                    logger.warning(f"用户{register_id}的图片文件夹为空")
                    continue

                image_path = os.path.join(image_folder, image_files[0])

                # 3. 提取新特征
                try:
                    image = Image.open(image_path).convert('RGB')
                    new_feature = self.fs.get_face_feature(image)

                    # 4. 替换原有特征文件
                    feature_file = f'../datas/faces/{i}.npy'
                    np.save(feature_file, new_feature)
                    logger.info(f"更新用户{register_id}的特征成功")

                except Exception as e:
                    logger.error(f"更新用户{register_id}的特征失败: {e}")
                    continue

            # 5. 清空并重新加载特征
            self.features = np.empty((0, 0))
            self.load_feature()

            QMessageBox.information(self, '提示', '人脸特征重新加载完成')

        except Exception as e:
            logger.error(f"重新加载人脸特征失败: {e}")
            QMessageBox.critical(self, '错误', f'重新加载人脸特征失败: {str(e)}')


    def video_open_close(self):
        """打开或关闭摄像头"""
        if self.open_flag:
            # 关闭摄像头
            self.open_flag = False
            self.button1.setText('打开摄像头')
            self.video.clear()

            # 确保摄像头资源释放
            if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
                self.cap.release()
                logger.info("Camera released in video_open_close")
        else:
            # 打开摄像头前确保之前的资源已释放
            if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
                self.cap.release()

            # 重新初始化摄像头
            try:
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if self.cap is not None:  # Check if cap was created successfully
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)

                    if self.cap.isOpened():
                        logger.info("Camera opened successfully in video_open_close")
                        self.open_flag = True
                        self.button1.setText('关闭摄像头')
                    else:
                        QMessageBox.warning(self, '警告', '无法打开摄像头')
                        logger.error("Failed to open camera in video_open_close")
                else:
                    QMessageBox.warning(self, '警告', '摄像头对象创建失败')
                    logger.error("Failed to create camera object in video_open_close")
            except Exception as e:
                QMessageBox.warning(self, '警告', f'打开摄像头时发生错误: {str(e)}')
                logger.error(f"Error opening camera in video_open_close: {e}")

    def get_max_boxes(self, faces: np.array):
        areas = (faces[:, 2] - faces[:, 0]) * (faces[:, 3] - faces[:, 1])
        max_index = np.argmax(areas)
        return faces[max_index, :]

    def display(self):
        """显示摄像头画面的线程函数"""
        try:
            while not self.is_shutting_down:
                if self.open_flag and hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
                    try:
                        w, h = self.video.width(), self.video.height()
                        # 读取帧
                        success, frame = self.cap.read()
                        if not success:
                            logger.warning("Failed to read frame from camera")
                            time.sleep(0.1)
                            continue

                        frame = cv2.resize(frame, (w - 20, h - 20), interpolation=cv2.INTER_AREA)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        image = Image.fromarray(frame, mode='RGB')
                        # frame大小是(w - 20, h - 20)给组件预留空白
                        frame = Image.fromarray(frame, mode='RGB')
                        # 重新设置格式
                        image = image.resize((512, 512))
                        # 人脸检测
                        faces = self.fs.face_detect(image)
                        faces = np.array(faces)
                        if faces.shape[0] > 0:
                            # 选择面积最大的box
                            faces = self.get_max_boxes(faces)
                            # 添加一个新的维度，归一化
                            faces = faces[np.newaxis, :]
                            faces = faces / 512
                            faces[:, 0] = faces[:, 0] * (w - 20)
                            faces[:, 1] = faces[:, 1] * (h - 20)
                            faces[:, 2] = faces[:, 2] * (w - 20)
                            faces[:, 3] = faces[:, 3] * (h - 20)
                            faces = faces.astype(int)
                            # 画出人脸框
                            frame = draw_bboxes(frame, faces)
                            with self.lock:  # 使用线程锁
                                self.faces = np.ascontiguousarray(faces)
                        self.frame = np.ascontiguousarray(np.array(frame))

                        # 将图像显示到界面
                        if self.open_flag and not self.is_shutting_down:
                            img = QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0],
                                         self.frame.shape[1] * 3, QImage.Format_RGB888)
                            self.video.setPixmap(QPixmap.fromImage(img))

                        # 控制帧率
                        cv2.waitKey(int(1000 / 30))

                    except Exception as e:
                        logger.error(f"Error in display loop: {e}")
                        time.sleep(0.1)
                else:
                    # 当摄像头关闭时降低CPU占用
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Display thread crashed: {e}")
        finally:
            logger.info("Display thread terminated")

    def closeEvent(self, event):
        """Handle window close event"""
        print("FaceWindow closeEvent triggered")
        # Prevent multiple closeEvent handling
        if not self.is_shutting_down:
            self.is_shutting_down = True
            self.shutdown_signal.emit()
            # Accept the event after cleanup signal is emitted
            event.accept()
        else:
            # Already shutting down, just accept the event
            event.accept()

    def on_app_quit(self):
        """Handle application quit signal"""
        print("Application quit signal received")
        if not self.is_shutting_down:
            self.is_shutting_down = True
            self.cleanup_resources()

    def cleanup_resources(self):
        """清理所有资源的函数"""
        print("Cleaning up resources...")

        # 设置标志停止显示线程
        self.open_flag = False
        self.is_shutting_down = True

        # 等待线程结束，使用较短的超时防止无限等待
        if hasattr(self, 'th') and self.th.is_alive():
            print("Waiting for display thread to finish...")
            # Shorter timeout to avoid blocking UI for too long
            self.th.join(timeout=0.5)
            print("Display thread finished or timeout reached")

        # 释放摄像头资源
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            print("Releasing camera...")
            self.cap.release()
            print("Camera released")

        # 关闭数据库连接
        if hasattr(self, 'connection') and self.connection:
            print("Closing database connection...")
            if hasattr(self, 'cursor') and self.cursor:
                self.cursor.close()
            self.connection.close()
            print("Database connection closed")

    def save_image(self, image, path):
        """
        将图片保存到指定路径
        :param image: 要保存的图片（numpy数组）
        :param path: 保存的路径
        """
        # 确保保存路径的目录存在
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 将numpy数组转换为BGR格式
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 保存图片
        cv2.imwrite(path, image)


if __name__ == '__main__':
    import sys
    import time

    # Set up logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    # Add exception hook to catch unhandled exceptions
    def exception_hook(exctype, value, traceback):
        print(f"Unhandled exception: {exctype.__name__}: {value}")
        sys.__excepthook__(exctype, value, traceback)


    sys.excepthook = exception_hook

    app = QApplication(sys.argv)
    window = FaceWindow()
    sys.exit(app.exec())