from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from config_loader import config

database_config = config["DATABASE"]
user_name = database_config["user_name"]
password = database_config["password"]
port = database_config["port"]
host = database_config["host"]
database_name = database_config["database_name"]

url = f"postgresql+psycopg2://{user_name}:{password}@{host}:{port}/{database_name}"
engine = create_engine(url)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
