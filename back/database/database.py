from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from database.engine_connector import connect_engine


engine = connect_engine()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
