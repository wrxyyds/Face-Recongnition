import pymysql
from datetime import datetime

try:
    # 建立数据库连接
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='88888888',
        database='face_reconginition',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

    # 获取当前时间
    now = datetime.now()
    punch_time = now.strftime('%Y-%m-%d %H:%M:%S')

    # 创建游标对象
    with connection.cursor() as cursor:
        # 插入数据的 SQL 语句
        sql = "INSERT INTO student_attendance (student_id, student_name, punch_time) VALUES (%s, %s, %s)"
        values = ('001', '张三', punch_time)

        # 执行插入操作
        cursor.execute(sql, values)

    # 提交事务
    connection.commit()
    print("数据插入成功！")

except pymysql.Error as e:
    # 回滚事务
    connection.rollback()
    print(f"插入数据时出错：{e}")
finally:
    # 关闭连接
    if connection:
        connection.close()
