import sys
import pymysql
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QTextEdit, QTabWidget, QPushButton,
                               QComboBox, QMessageBox, QTableWidget, QTableWidgetItem,
                               QHeaderView, QGroupBox, QLineEdit, QDateEdit, QTimeEdit)
from PySide6.QtCore import Qt, QDate, QTime, QDateTime
from PySide6.QtGui import QFont
from datetime import date, datetime


class AttendanceManagementSystem(QWidget):
    def __init__(self):
        super().__init__()
        self.students = {}
        self.teachers = {}
        self.load_user_data()
        self.initUI()

    def load_user_data(self):
        """加载用户数据"""
        try:
            with open('./datas/names.txt', 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        parts = line.split(',')
                        if len(parts) >= 3:
                            register_id, name, identity = parts[0], parts[1], parts[2]
                            if identity == 'student':
                                self.students[register_id] = name
                            elif identity == 'teacher':
                                self.teachers[register_id] = name
        except FileNotFoundError:
            QMessageBox.warning(self, "警告", "找不到用户数据文件 './datas/names.txt'")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载用户数据时出错: {str(e)}")

    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle('考勤管理系统')
        self.setGeometry(100, 100, 1000, 700)

        # 设置字体
        font = QFont()
        font.setPointSize(10)
        self.setFont(font)

        # 创建主标签页
        tab_widget = QTabWidget()
        tab_widget.addTab(self.create_statistics_tab(), '统计考勤记录')
        tab_widget.addTab(self.create_management_tab(), '教师考勤管理')

        main_layout = QVBoxLayout()
        main_layout.addWidget(tab_widget)
        self.setLayout(main_layout)

        # 加载初始数据
        self.load_attendance_statistics()

    def create_statistics_tab(self):
        """创建统计考勤记录标签页"""
        tab = QWidget()
        layout = QVBoxLayout()

        # 刷新按钮
        refresh_btn = QPushButton('刷新数据')
        refresh_btn.clicked.connect(self.load_attendance_statistics)
        layout.addWidget(refresh_btn)

        # 统计信息显示区域
        stats_group = QGroupBox("考勤统计")
        stats_layout = QVBoxLayout()

        self.stats_label = QLabel()
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        stats_layout.addWidget(self.stats_label)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # 创建考勤详情表格
        self.attendance_table = QTableWidget()
        self.attendance_table.setColumnCount(4)
        self.attendance_table.setHorizontalHeaderLabels(['学号', '姓名', '打卡时间', '状态'])

        # 设置表格属性
        header = self.attendance_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.attendance_table.setAlternatingRowColors(True)
        self.attendance_table.setSelectionBehavior(QTableWidget.SelectRows)

        layout.addWidget(QLabel("考勤详情:"))
        layout.addWidget(self.attendance_table)

        tab.setLayout(layout)
        return tab

    def create_management_tab(self):
        """创建教师考勤管理标签页"""
        tab = QWidget()
        layout = QVBoxLayout()

        # 添加考勤记录区域
        add_group = QGroupBox("添加考勤记录")
        add_layout = QVBoxLayout()

        # 学生选择
        student_layout = QHBoxLayout()
        student_layout.addWidget(QLabel("选择学生:"))
        self.student_combo = QComboBox()
        self.student_combo.addItems([f"{sid} - {name}" for sid, name in self.students.items()])
        student_layout.addWidget(self.student_combo)
        add_layout.addLayout(student_layout)

        # 日期时间选择
        datetime_layout = QHBoxLayout()
        datetime_layout.addWidget(QLabel("日期:"))
        self.date_edit = QDateEdit()
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        datetime_layout.addWidget(self.date_edit)

        datetime_layout.addWidget(QLabel("时间:"))
        self.time_edit = QTimeEdit()
        self.time_edit.setTime(QTime.currentTime())
        datetime_layout.addWidget(self.time_edit)
        add_layout.addLayout(datetime_layout)

        # 添加按钮
        add_btn = QPushButton('添加考勤记录')
        add_btn.clicked.connect(self.add_attendance_record)
        add_layout.addWidget(add_btn)

        add_group.setLayout(add_layout)
        layout.addWidget(add_group)

        # 删除考勤记录区域
        delete_group = QGroupBox("删除考勤记录")
        delete_layout = QVBoxLayout()

        # 已签到学生列表
        self.signed_students_table = QTableWidget()
        self.signed_students_table.setColumnCount(4)
        self.signed_students_table.setHorizontalHeaderLabels(['学号', '姓名', '打卡时间', '操作'])

        header = self.signed_students_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.signed_students_table.setAlternatingRowColors(True)

        delete_layout.addWidget(QLabel("今日已签到学生:"))
        delete_layout.addWidget(self.signed_students_table)

        # 刷新按钮
        refresh_signed_btn = QPushButton('刷新已签到列表')
        refresh_signed_btn.clicked.connect(self.load_signed_students)
        delete_layout.addWidget(refresh_signed_btn)

        delete_group.setLayout(delete_layout)
        layout.addWidget(delete_group)

        # 初始加载已签到学生
        self.load_signed_students()

        tab.setLayout(layout)
        return tab

    def get_db_connection(self):
        """获取数据库连接"""
        try:
            return pymysql.connect(
                host='localhost',
                user='root',
                password='88888888',
                database='face_reconginition',
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
        except Exception as e:
            QMessageBox.critical(self, "数据库连接错误", f"无法连接到数据库: {str(e)}")
            return None

    def load_attendance_statistics(self):
        """加载考勤统计数据"""
        connection = self.get_db_connection()
        if not connection:
            return

        today = date.today().strftime('%Y-%m-%d')

        try:
            with connection.cursor() as cursor:
                # 获取今天已签到学生信息
                cursor.execute("""
                    SELECT student_id, student_name, MAX(punch_time) as punch_time
                    FROM student_attendance
                    WHERE DATE(punch_time) = %s
                    GROUP BY student_id, student_name
                    ORDER BY punch_time DESC
                """, (today,))
                signed_students = cursor.fetchall()
                signed_student_ids = [student['student_id'] for student in signed_students]

                # 获取未签到学生
                unsigned_students = [(sid, name) for sid, name in self.students.items()
                                     if sid not in signed_student_ids]

                # 计算签到率
                total_students = len(self.students)
                signed_count = len(signed_students)
                attendance_rate = (signed_count / total_students * 100) if total_students > 0 else 0

                # 更新统计标签
                stats_text = f"总学生数: {total_students} | 已签到: {signed_count} | 未签到: {len(unsigned_students)} | 签到率: {attendance_rate:.1f}%"
                self.stats_label.setText(stats_text)

                # 更新表格
                self.update_attendance_table(signed_students, unsigned_students)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载考勤统计时出错: {str(e)}")
        finally:
            connection.close()

    def update_attendance_table(self, signed_students, unsigned_students):
        """更新考勤表格"""
        total_rows = len(signed_students) + len(unsigned_students)
        self.attendance_table.setRowCount(total_rows)

        row = 0
        # 添加已签到学生
        for student in signed_students:
            self.attendance_table.setItem(row, 0, QTableWidgetItem(student['student_id']))
            self.attendance_table.setItem(row, 1, QTableWidgetItem(student['student_name']))
            self.attendance_table.setItem(row, 2, QTableWidgetItem(str(student['punch_time'])))
            status_item = QTableWidgetItem("已签到")
            status_item.setBackground(Qt.green)
            self.attendance_table.setItem(row, 3, status_item)
            row += 1

        # 添加未签到学生
        for student_id, name in unsigned_students:
            self.attendance_table.setItem(row, 0, QTableWidgetItem(student_id))
            self.attendance_table.setItem(row, 1, QTableWidgetItem(name))
            self.attendance_table.setItem(row, 2, QTableWidgetItem("--"))
            status_item = QTableWidgetItem("未签到")
            status_item.setBackground(Qt.red)
            self.attendance_table.setItem(row, 3, status_item)
            row += 1

    def add_attendance_record(self):
        """添加考勤记录"""
        if not self.student_combo.currentText():
            QMessageBox.warning(self, "警告", "请选择学生")
            return

        # 获取选择的学生ID
        student_text = self.student_combo.currentText()
        student_id = student_text.split(' - ')[0]
        student_name = self.students[student_id]

        # 获取日期时间
        selected_date = self.date_edit.date().toPython()
        selected_time = self.time_edit.time().toPython()
        punch_datetime = datetime.combine(selected_date, selected_time)

        connection = self.get_db_connection()
        if not connection:
            return

        try:
            with connection.cursor() as cursor:
                # 检查是否已经有今天的签到记录
                cursor.execute("""
                    SELECT COUNT(*) as count FROM student_attendance 
                    WHERE student_id = %s AND DATE(punch_time) = %s
                """, (student_id, selected_date.strftime('%Y-%m-%d')))

                result = cursor.fetchone()
                if result['count'] > 0:
                    reply = QMessageBox.question(self, "确认",
                                                 f"学生 {student_name} 今天已有签到记录，是否继续添加？",
                                                 QMessageBox.Yes | QMessageBox.No)
                    if reply == QMessageBox.No:
                        return

                # 插入考勤记录
                cursor.execute("""
                    INSERT INTO student_attendance (student_id, student_name, punch_time)
                    VALUES (%s, %s, %s)
                """, (student_id, student_name, punch_datetime))

                connection.commit()
                QMessageBox.information(self, "成功", f"已为学生 {student_name} 添加考勤记录")

                # 刷新数据
                self.load_attendance_statistics()
                self.load_signed_students()

        except Exception as e:
            connection.rollback()
            QMessageBox.critical(self, "错误", f"添加考勤记录时出错: {str(e)}")
        finally:
            connection.close()

    def load_signed_students(self):
        """加载已签到学生列表"""
        connection = self.get_db_connection()
        if not connection:
            return

        today = date.today().strftime('%Y-%m-%d')

        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT id, student_id, student_name, punch_time
                    FROM student_attendance
                    WHERE DATE(punch_time) = %s
                    ORDER BY punch_time DESC
                """, (today,))

                signed_students = cursor.fetchall()

                # 更新表格
                self.signed_students_table.setRowCount(len(signed_students))

                for row, student in enumerate(signed_students):
                    self.signed_students_table.setItem(row, 0, QTableWidgetItem(student['student_id']))
                    self.signed_students_table.setItem(row, 1, QTableWidgetItem(student['student_name']))
                    self.signed_students_table.setItem(row, 2, QTableWidgetItem(str(student['punch_time'])))

                    # 添加删除按钮
                    delete_btn = QPushButton('删除')
                    delete_btn.clicked.connect(
                        lambda checked, record_id=student['id']: self.delete_attendance_record(record_id))
                    self.signed_students_table.setCellWidget(row, 3, delete_btn)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载已签到学生时出错: {str(e)}")
        finally:
            connection.close()

    def delete_attendance_record(self, record_id):
        """删除考勤记录"""
        reply = QMessageBox.question(self, "确认删除",
                                     "确定要删除这条考勤记录吗？",
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.No:
            return

        connection = self.get_db_connection()
        if not connection:
            return

        try:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM student_attendance WHERE id = %s", (record_id,))
                connection.commit()

                QMessageBox.information(self, "成功", "考勤记录已删除")

                # 刷新数据
                self.load_attendance_statistics()
                self.load_signed_students()

        except Exception as e:
            connection.rollback()
            QMessageBox.critical(self, "错误", f"删除考勤记录时出错: {str(e)}")
        finally:
            connection.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle('Fusion')

    window = AttendanceManagementSystem()
    window.show()

    sys.exit(app.exec())