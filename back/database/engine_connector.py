from sqlalchemy import create_engine
import configparser


def connect_engine():
    config = configparser.ConfigParser()
    config.read('./config.ini')

    database_config = config['DATABASE']
    user_name = database_config['user_name']
    password = database_config['password']
    port = database_config['port']
    host = database_config['host']
    database_name = database_config['database_name']

    url = f'postgresql+psycopg2://{user_name}:{password}@{host}:{port}/{database_name}'
    engine = create_engine(url)

    return engine