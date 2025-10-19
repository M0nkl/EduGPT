import sqlite3
import os
from datetime import datetime


def create_database():
    if not os.path.exists('data'):
        os.makedirs('data')

    conn = sqlite3.connect('data/methodics.db')
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS methodics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        subject TEXT,
        author TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Добавляем тестовые данные
    test_data = [
        (
            "Методика преподавания математики в школе",
            "В данном пособии рассматриваются современные подходы к преподаванию математики...",
            "Математика",
            "Иванов И.И."
        ),
        (
            "Дифференциальные уравнения для начинающих",
            "Пособие по дифференциальным уравнениям с примерами и задачами для студентов...",
            "Математика",
            "Петров П.П."
        ),
        (
            "Основы программирования на Python",
            "Методическое пособие по основам программирования на языке Python...",
            "Информатика",
            "Сидоров С.С."
        ),
        (
            "Методика обучения физике",
            "Современные методы преподавания физики в средней школе...",
            "Физика",
            "Кузнецов К.К."
        ),
        (
            "Английский язык для технических специальностей",
            "Методическое пособие по английскому языку для студентов технических вузов...",
            "Английский язык",
            "Смирнова А.А."
        )
    ]

    cursor.executemany('''
    INSERT INTO methodics (title, content, subject, author)
    VALUES (?, ?, ?, ?)
    ''', test_data)

    # Сохраняем изменения и закрываем соединение
    conn.commit()
    conn.close()

    print("База данных успешно создана!")
    print("Добавлено 5 тестовых методичек")


if __name__ == "__main__":
    create_database()