import os
from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.engine import URL

load_dotenv()  # 加载 .env 文件中的环境变量 

db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")


DATABASE_URL = URL.create(
    drivername="postgresql+psycopg",
    username=db_user,
    password=db_password,
    host=db_host,
    port=int(db_port),
    database=db_name
)

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