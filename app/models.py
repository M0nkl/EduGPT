from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Methodic(Base):
    __tablename__ = "methodics"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content = Column(Text)
    subject = Column(String)
    author = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Methodic {self.title}>"