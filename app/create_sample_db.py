from database import init_db, SessionLocal
from models import Methodic

def create_sample_data():
    """Создает базу данных с примерными данными"""
    init_db()
    print("База данных создана с примерными методичками!")

if __name__ == "__main__":
    create_sample_data()