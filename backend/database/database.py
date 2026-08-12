from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base


DATABASE_URL = "sqlite:///./test.db"


engine = create_engine(
    DATABASE_URL
)


SessionLocal = sessionmaker(
    bind=engine
)


Base = declarative_base()


# 提供数据库会话对象，供依赖注入使用
def get_db():

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()