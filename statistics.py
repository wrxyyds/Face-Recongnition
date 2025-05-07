import sys
import pymysql
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QTabWidget
from PySide6.QtCore import Qt
from datetime import date

class StatisticsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.students = {}
        self.teachers = {}

        with open('./datas/names.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip()
                register_id, name, identity = line.split(',')
                if identity == 'student':
                    self.students[register_id] = name
                if identity == 'teacher':
                    self.teachers[register_id] = name
        self.initUI()
        self.load_attendance_data()

    def initUI(self):
        self.setWindowTitle('考勤统计信息')
        self.setGeometry(100, 100, 800, 600)

        # 创建一个标签页窗口
        tab_widget = QTabWidget()

        # 创建学生考勤信息标签页
        self.student_tab = QWidget()
        self.student_layout = QVBoxLayout()
        self.student_attendance_text = QTextEdit()
        self.student_attendance_text.setReadOnly(True)
        self.student_layout.addWidget(self.student_attendance_text)
        self.student_tab.setLayout(self.student_layout)
        tab_widget.addTab(self.student_tab, '学生考勤信息')

        # 创建教师考勤信息标签页
        self.teacher_tab = QWidget()
        self.teacher_layout = QVBoxLayout()
        self.teacher_attendance_text = QTextEdit()
        self.teacher_attendance_text.setReadOnly(True)
        self.teacher_layout.addWidget(self.teacher_attendance_text)
        self.teacher_tab.setLayout(self.teacher_layout)
        tab_widget.addTab(self.teacher_tab, '教师考勤信息')

        layout = QVBoxLayout()
        layout.addWidget(tab_widget)
        self.setLayout(layout)

    def load_attendance_data(self):
        # 建立数据库连接
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='88888888',
            database='face_reconginition',
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

        today = date.today().strftime('%Y-%m-%d')

        try:
            with connection.cursor() as cursor:
                # 获取今天已签到学生信息，相同学号取时间最晚的
                cursor.execute("""
                    SELECT student_id, student_name, MAX(punch_time) as punch_time
                    FROM student_attendance
                    WHERE DATE(punch_time) = %s
                    GROUP BY student_id, student_name
                """, (today,))
                signed_students = cursor.fetchall()
                signed_student_ids = [student['student_id'] for student in signed_students]

                # 获取所有学生信息
                all_students = self.students
                unsigned_students = [(student_id, name) for student_id, name in all_students.items() if
                                     student_id not in signed_student_ids]

                # 计算学生签到率
                total_students = len(all_students)
                student_attendance_rate = len(signed_students) / total_students if total_students > 0 else 0

                # 显示学生考勤信息
                student_info = f"学生签到率: {student_attendance_rate * 100:.2f}%\n\n"
                student_info += "已签到学生信息:\n"
                for student in signed_students:
                    student_info += f"学号: {student['student_id']}, 姓名: {student['student_name']}, 打卡时间: {student['punch_time']}\n"
                student_info += "\n未签到学生信息:\n"
                for student_id, name in unsigned_students:
                    student_info += f"学号: {student_id}, 姓名: {name}\n"
                self.student_attendance_text.setPlainText(student_info)

                # 获取今天已签到教师信息，相同学号取时间最晚的
                cursor.execute("""
                    SELECT teacher_id, teacher_name, MAX(punch_time) as punch_time
                    FROM teacher_attendance
                    WHERE DATE(punch_time) = %s
                    GROUP BY teacher_id, teacher_name
                """, (today,))
                signed_teachers = cursor.fetchall()
                signed_teacher_ids = [teacher['teacher_id'] for teacher in signed_teachers]

                # 获取所有教师信息
                all_teachers = self.teachers
                unsigned_teachers = [(teacher_id, name) for teacher_id, name in all_teachers.items() if
                                     teacher_id not in signed_teacher_ids]

                # 计算教师签到率
                total_teachers = len(all_teachers)
                teacher_attendance_rate = len(signed_teachers) / total_teachers if total_teachers > 0 else 0

                # 显示教师考勤信息
                teacher_info = f"教师签到率: {teacher_attendance_rate * 100:.2f}%\n\n"
                teacher_info += "已签到教师信息:\n"
                for teacher in signed_teachers:
                    teacher_info += f"工号: {teacher['teacher_id']}, 姓名: {teacher['teacher_name']}, 打卡时间: {teacher['punch_time']}\n"
                teacher_info += "\n未签到教师信息:\n"
                for teacher_id, name in unsigned_teachers:
                    teacher_info += f"工号: {teacher_id}, 姓名: {name}\n"
                self.teacher_attendance_text.setPlainText(teacher_info)

        except Exception as e:
            print(f"数据库查询出错: {e}")
        finally:
            connection.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = StatisticsWindow()
    window.show()
    sys.exit(app.exec())