# init_db.py
from database import engine
from models import Base

def init_database():
    """Инициализация базы данных, создание таблиц"""
    Base.metadata.create_all(bind=engine)
    print("Таблицы созданы: methodic_entries, qa_entries")

if __name__ == "__main__":
    init_database()