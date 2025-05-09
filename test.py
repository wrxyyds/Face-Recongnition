import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import cv2
import numpy as np
from datetime import datetime
from face.facenet.model import Resnet34Triplet


class FaceRegistrationSystem:
    def __init__(self, base_model_path='../face/facenet/weights/model_resnet34_triplet.pt',
                 fine_tuned_model_path='../face/facenet/weights/transferred_facenet_model.pt',
                 train_img_dir='../images/train/',
                 embedding_dim=512,
                 device=None):
        """
        初始化人脸注册系统

        Args:
            base_model_path: 基础预训练模型路径
            fine_tuned_model_path: 微调后模型保存路径
            train_img_dir: 训练图像目录
            embedding_dim: 嵌入向量维度
            device: 计算设备 (CPU/GPU)
        """
        # 设置路径和配置
        self.base_model_path = base_model_path
        self.fine_tuned_model_path = fine_tuned_model_path
        self.train_img_dir = train_img_dir
        self.embedding_dim = embedding_dim

        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"使用设备: {self.device}")

        # 初始化模型
        self.model = self._initialize_model()

        # 设置图像变换
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.6071, 0.4609, 0.3944],
                std=[0.2457, 0.2175, 0.2129]
            )
        ])

        # 评估用变换（无数据增强）
        self.eval_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.6071, 0.4609, 0.3944],
                std=[0.2457, 0.2175, 0.2129]
            )
        ])

    def _initialize_model(self):
        """初始化模型，优先加载已经微调过的模型"""
        model = Resnet34Triplet(embedding_dimension=self.embedding_dim)

        if os.path.exists(self.fine_tuned_model_path):
            print(f"加载已微调模型: {self.fine_tuned_model_path}")
            checkpoint = torch.load(self.fine_tuned_model_path, map_location=self.device)
            model.load_state_dict(checkpoint)
        else:
            print(f"加载基础预训练模型: {self.base_model_path}")
            checkpoint = torch.load(self.base_model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

        model.to(self.device)
        return model

    def capture_student_faces(self, student_id, num_images=5, camera_index=0):
        """
        使用摄像头捕获学生人脸图像

        Args:
            student_id: 学生ID
            num_images: 要捕获的图像数量
            camera_index: 摄像头索引

        Returns:
            bool: 捕获是否成功
        """
        # 创建学生图像目录
        student_dir = os.path.join(self.train_img_dir, student_id)
        if not os.path.exists(student_dir):
            os.makedirs(student_dir)

        # 初始化摄像头
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("无法打开摄像头")
            return False

        # 加载人脸检测器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        captured_count = 0
        while captured_count < num_images:
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                print("无法获取图像帧")
                break

            # 转换为灰度进行人脸检测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # 显示图像和检测到的人脸
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow('Face Registration', frame)

            # 每隔一段时间自动捕获或按空格键手动捕获
            key = cv2.waitKey(100)
            if key == 32 or (captured_count == 0 and len(faces) > 0):  # 空格键或自动捕获第一张
                # 如果检测到人脸
                if len(faces) > 0:
                    x, y, w, h = faces[0]  # 取第一个检测到的人脸
                    # 扩大人脸区域以包含更多上下文
                    margin = int(0.2 * max(w, h))
                    x_start = max(0, x - margin)
                    y_start = max(0, y - margin)
                    x_end = min(frame.shape[1], x + w + margin)
                    y_end = min(frame.shape[0], y + h + margin)

                    face_img = frame[y_start:y_end, x_start:x_end]

                    # 保存图像
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_path = os.path.join(student_dir, f"{student_id}_{timestamp}_{captured_count}.jpg")
                    cv2.imwrite(img_path, face_img)

                    captured_count += 1
                    print(f"已捕获 {captured_count}/{num_images} 张人脸图像")

            # 按ESC键退出
            if key == 27:
                break

        # 释放资源
        cap.release()
        cv2.destroyAllWindows()

        if captured_count > 0:
            print(f"成功为学生 {student_id} 捕获了 {captured_count} 张人脸图像")
            # 生成额外的数据增强样本
            self._generate_augmented_samples(student_dir)
            return True
        else:
            print(f"未能捕获学生 {student_id} 的人脸图像")
            return False

    def _generate_augmented_samples(self, student_dir, min_samples=6):
        """为学生生成额外的增强样本"""
        # 获取目录中的图像文件
        img_files = [f for f in os.listdir(student_dir)
                     if f.endswith(('.jpg', '.jpeg', '.png'))]

        # 如果样本数量已经足够，则不需要生成
        if len(img_files) >= min_samples:
            return

        # 定义增强变换
        augmentations = [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.GaussianBlur(3, sigma=(0.1, 0.5))
        ]

        # 为每个原始图像生成新的增强版本
        for img_file in img_files[:]:  # 使用原始列表副本
            img_path = os.path.join(student_dir, img_file)
            img = Image.open(img_path).convert('RGB')

            # 生成文件名不包含扩展名的部分
            file_base = os.path.splitext(img_file)[0]

            # 应用每个增强并保存新图像
            for i, transform in enumerate(augmentations):
                augmented_img = transform(img)
                aug_filename = f"{file_base}_aug{i}.jpg"
                aug_path = os.path.join(student_dir, aug_filename)
                augmented_img.save(aug_path)

        print(f"为 {student_dir} 生成了增强样本")

    def register_student(self, student_id, capture_images=True, num_images=5, retrain=True):
        """
        注册新学生

        Args:
            student_id: 学生ID
            capture_images: 是否捕获图像
            num_images: 要捕获的图像数量
            retrain: 是否立即重新训练模型

        Returns:
            bool: 注册是否成功
        """
        # 检查学生是否已存在
        student_dir = os.path.join(self.train_img_dir, student_id)
        if os.path.exists(student_dir):
            print(f"学生 {student_id} 已存在于系统中")
            # 可以选择更新或覆盖
            response = input("是否更新此学生的人脸数据? (y/n): ")
            if response.lower() != 'y':
                return False
        else:
            os.makedirs(student_dir)

        # 捕获人脸图像
        if capture_images:
            success = self.capture_student_faces(student_id, num_images)
            if not success:
                return False

        # 检查是否有足够的图像进行训练
        img_files = [f for f in os.listdir(student_dir)
                     if f.endswith(('.jpg', '.jpeg', '.png'))]
        if len(img_files) < 2:
            print(f"学生 {student_id} 的图像数量不足，至少需要2张图像")
            return False

        # 重新训练模型
        if retrain:
            success = self.retrain_model(student_id)
            if not success:
                print("模型重新训练失败")
                return False

        print(f"学生 {student_id} 注册成功")
        return True

    def retrain_model(self, target_student_id, num_epochs=20, batch_size=8, learning_rate=0.001):
        """
        重新训练模型，将新学生的人脸嵌入到模型中

        Args:
            target_student_id: 目标学生ID
            num_epochs: 训练轮次
            batch_size: 批次大小
            learning_rate: 学习率

        Returns:
            bool: 训练是否成功
        """
        print(f"开始为学生 {target_student_id} 重新训练模型...")

        try:
            # 创建数据集
            from train import EnhancedFaceDataset

            dataset = EnhancedFaceDataset(
                root_dir=self.train_img_dir,
                target_cls=target_student_id,
                transform=self.transform,
                use_hard_triplets=True,
                model=self.model
            )

            # 检查数据集大小
            if len(dataset) == 0:
                print("数据集为空，无法训练")
                return False

            # 设置优化器
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

            # 使用学习率调度器
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,
                                          verbose=True, min_lr=1e-5)

            # 三元组损失
            criterion = nn.TripletMarginLoss(margin=0.3)

            # 训练参数
            patience = 3  # 早停参数
            best_loss = float('inf')
            patience_counter = 0

            # 训练循环
            for epoch in range(num_epochs):
                # 刷新三元组
                if epoch > 0:
                    dataset.model = self.model
                    dataset.refresh_triplets()

                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                # 训练一个轮次
                self.model.train()
                running_loss = 0.0
                batch_count = 0

                for i, (anchor, positive, negative) in enumerate(dataloader):
                    # 将数据移至设备
                    anchor = anchor.to(self.device)
                    positive = positive.to(self.device)
                    negative = negative.to(self.device)

                    # 清零参数梯度
                    optimizer.zero_grad()

                    # 前向传播
                    anchor_embedding = self.model(anchor)
                    positive_embedding = self.model(positive)
                    negative_embedding = self.model(negative)

                    # 计算三元组损失
                    loss = criterion(anchor_embedding, positive_embedding, negative_embedding)

                    # 反向传播和优化
                    loss.backward()
                    optimizer.step()

                    # 统计
                    running_loss += loss.item()
                    batch_count += 1

                # 计算平均损失
                epoch_loss = running_loss / batch_count if batch_count > 0 else float('inf')
                print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

                # 更新学习率调度器
                scheduler.step(epoch_loss)

                # 早停检查
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    # 保存最佳模型
                    torch.save(self.model.state_dict(), self.fine_tuned_model_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

            print("模型训练完成")
            # 重新加载最佳模型
            self.model.load_state_dict(torch.load(self.fine_tuned_model_path,
                                                  map_location=self.device))
            return True

        except Exception as e:
            print(f"训练过程中发生错误: {str(e)}")
            return False

    def get_face_embeddings(self, img_path):
        """
        获取人脸嵌入向量

        Args:
            img_path: 图像路径

        Returns:
            numpy.ndarray: 嵌入向量
        """
        try:
            self.model.eval()
            img = Image.open(img_path).convert('RGB')
            img = self.eval_transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model(img)

            return embedding.cpu().numpy()

        except Exception as e:
            print(f"获取嵌入向量时出错: {str(e)}")
            return None

    def compute_all_embeddings(self):
        """
        计算所有学生的嵌入向量

        Returns:
            dict: 学生ID到嵌入向量的映射
        """
        embeddings = {}
        self.model.eval()

        for student_id in os.listdir(self.train_img_dir):
            student_dir = os.path.join(self.train_img_dir, student_id)
            if not os.path.isdir(student_dir):
                continue

            student_embeddings = []
            img_files = [f for f in os.listdir(student_dir)
                         if f.endswith(('.jpg', '.jpeg', '.png'))]

            for img_file in img_files:
                img_path = os.path.join(student_dir, img_file)
                embedding = self.get_face_embeddings(img_path)
                if embedding is not None:
                    student_embeddings.append(embedding.squeeze())

            if student_embeddings:
                # 计算平均嵌入向量
                embeddings[student_id] = np.mean(student_embeddings, axis=0)

        return embeddings

    def identify_student(self, img_path, threshold=0.7):
        """
        识别学生

        Args:
            img_path: 图像路径
            threshold: 识别阈值

        Returns:
            tuple: (学生ID, 相似度)，如果没有识别到则返回(None, 0)
        """
        # 获取图像嵌入向量
        query_embedding = self.get_face_embeddings(img_path)
        if query_embedding is None:
            return None, 0

        query_embedding = query_embedding.squeeze()

        # 获取所有学生的嵌入向量
        all_embeddings = self.compute_all_embeddings()

        best_match = None
        best_similarity = 0

        # 计算相似度并找到最佳匹配
        for student_id, embedding in all_embeddings.items():
            # 计算余弦相似度
            similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding))

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = student_id

        # 检查是否超过阈值
        if best_similarity >= threshold:
            return best_match, best_similarity
        else:
            return None, best_similarity


class AttendanceSystem:
    def __init__(self, face_system, attendance_log_path='../logs/attendance.csv'):
        """
        初始化考勤系统

        Args:
            face_system: 人脸识别系统
            attendance_log_path: 考勤日志路径
        """
        self.face_system = face_system
        self.attendance_log_path = attendance_log_path

        # 确保日志目录存在
        os.makedirs(os.path.dirname(attendance_log_path), exist_ok=True)

        # 如果日志文件不存在，创建它并写入表头
        if not os.path.exists(attendance_log_path):
            with open(attendance_log_path, 'w') as f:
                f.write("student_id,timestamp,status\n")

    def take_attendance(self, camera_index=0, threshold=0.7, time_interval=5):
        """
        进行考勤

        Args:
            camera_index: 摄像头索引
            threshold: 识别阈值
            time_interval: 同一学生的最小识别间隔（秒）
        """
        print("开始考勤...")

        # 初始化摄像头
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("无法打开摄像头")
            return

        # 加载人脸检测器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 记录已识别的学生以避免重复记录
        recognized_students = {}

        while True:
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                print("无法获取图像帧")
                break

            # 转换为灰度进行人脸检测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            current_time = datetime.now()

            for (x, y, w, h) in faces:
                # 绘制人脸框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # 扩大人脸区域以包含更多上下文
                margin = int(0.2 * max(w, h))
                x_start = max(0, x - margin)
                y_start = max(0, y - margin)
                x_end = min(frame.shape[1], x + w + margin)
                y_end = min(frame.shape[0], y + h + margin)

                face_img = frame[y_start:y_end, x_start:x_end]

                # 保存临时图像用于识别
                temp_img_path = "../temp/temp_face.jpg"
                os.makedirs(os.path.dirname(temp_img_path), exist_ok=True)
                cv2.imwrite(temp_img_path, face_img)

                # 识别学生
                student_id, similarity = self.face_system.identify_student(temp_img_path, threshold)

                # 显示识别结果
                if student_id:
                    # 检查是否应该记录考勤（避免短时间内重复记录）
                    should_record = True
                    if student_id in recognized_students:
                        time_diff = (current_time - recognized_students[student_id]).total_seconds()
                        if time_diff < time_interval:
                            should_record = False

                    # 显示识别信息
                    info_text = f"ID: {student_id}, 相似度: {similarity:.2f}"
                    cv2.putText(frame, info_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # 记录考勤
                    if should_record:
                        self._record_attendance(student_id)
                        recognized_students[student_id] = current_time
                        print(f"记录学生 {student_id} 的考勤")
                else:
                    # 显示未识别信息
                    cv2.putText(frame, f"未识别, 相似度: {similarity:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2)

            # 显示图像
            cv2.imshow('考勤系统', frame)

            # 按ESC键退出
            if cv2.waitKey(1) == 27:
                break

        # 释放资源
        cap.release()
        cv2.destroyAllWindows()

        print("考勤结束")

    def _record_attendance(self, student_id, status="present"):
        """记录考勤"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.attendance_log_path, 'a') as f:
            f.write(f"{student_id},{timestamp},{status}\n")

    def generate_report(self, date=None, class_id=None):
        """
        生成考勤报告

        Args:
            date: 日期字符串 (YYYY-MM-DD)，如果为None则为当天
            class_id: 班级ID，如果为None则为所有班级

        Returns:
            dict: 考勤统计
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        print(f"生成 {date} 的考勤报告...")

        attendance_data = {}

        # 读取考勤日志
        try:
            with open(self.attendance_log_path, 'r') as f:
                lines = f.readlines()[1:]  # 跳过表头

                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        student_id, timestamp, status = parts[0], parts[1], parts[2]
                        log_date = timestamp.split(' ')[0]

                        # 检查日期匹配
                        if log_date == date:
                            # 检查班级匹配（如果指定）
                            if class_id is None or student_id.startswith(class_id):
                                if student_id not in attendance_data:
                                    attendance_data[student_id] = {
                                        "first_time": timestamp,
                                        "status": status
                                    }
        except Exception as e:
            print(f"读取考勤日志时出错: {str(e)}")

        # 生成统计信息
        total_students = len(os.listdir(self.face_system.train_img_dir))
        present_students = len(attendance_data)
        absent_students = total_students - present_students

        report = {
            "date": date,
            "total_students": total_students,
            "present_students": present_students,
            "absent_students": absent_students,
            "attendance_rate": present_students / total_students if total_students > 0 else 0,
            "details": attendance_data
        }

        print(f"考勤率: {report['attendance_rate']:.2%}")
        return report


# 使用示例
if __name__ == "__main__":
    # 创建人脸注册系统
    face_system = FaceRegistrationSystem()

    # 注册新学生
    student_id = input("请输入学生ID: ")
    face_system.register_student(student_id)

    # 创建考勤系统
    attendance_system = AttendanceSystem(face_system)

    # 进行考勤
    attendance_system.take_attendance()

    # 生成报告
    report = attendance_system.generate_report()
    print(report)