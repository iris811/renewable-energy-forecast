"""
Database configuration and models for renewable energy forecast system
"""
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import yaml
import os

Base = declarative_base()


class WeatherData(Base):
    """Weather data table"""
    __tablename__ = 'weather_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    location_name = Column(String(100), nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

    # Weather parameters
    temperature = Column(Float)  # Celsius
    humidity = Column(Float)  # %
    pressure = Column(Float)  # hPa
    wind_speed = Column(Float)  # m/s
    wind_direction = Column(Float)  # degrees
    cloud_cover = Column(Float)  # %
    precipitation = Column(Float)  # mm

    # Solar-specific
    solar_irradiance = Column(Float)  # W/m²
    ghi = Column(Float)  # Global Horizontal Irradiance
    dni = Column(Float)  # Direct Normal Irradiance
    dhi = Column(Float)  # Diffuse Horizontal Irradiance

    # Metadata
    source = Column(String(50))  # API source
    created_at = Column(DateTime, default=datetime.utcnow)


class PowerGeneration(Base):
    """Actual power generation data"""
    __tablename__ = 'power_generation'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    location_name = Column(String(100), nullable=False, index=True)

    # Power data
    power_output = Column(Float, nullable=False)  # kW
    energy_type = Column(String(20), nullable=False)  # 'solar' or 'wind'
    capacity = Column(Float)  # Total capacity in kW
    capacity_factor = Column(Float)  # Actual/Capacity

    # Metadata
    is_predicted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    """Model predictions"""
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_timestamp = Column(DateTime, nullable=False, index=True)
    target_timestamp = Column(DateTime, nullable=False, index=True)
    location_name = Column(String(100), nullable=False)

    # Prediction data
    predicted_power = Column(Float, nullable=False)
    energy_type = Column(String(20), nullable=False)
    confidence_interval_lower = Column(Float)
    confidence_interval_upper = Column(Float)

    # Model info
    model_name = Column(String(50))
    model_version = Column(String(20))

    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Database connection and session manager"""

    def __init__(self, config_path='configs/config.yaml'):
        self.config = self._load_config(config_path)
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(bind=self.engine)

    def _load_config(self, config_path):
        """Load database configuration"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return None

    def _create_engine(self):
        """Create database engine"""
        if self.config and self.config.get('database'):
            db_config = self.config['database']
            db_type = db_config.get('type', 'sqlite')

            if db_type == 'sqlite':
                db_path = db_config['sqlite']['path']
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                connection_string = f'sqlite:///{db_path}'
            elif db_type == 'postgresql':
                pg_config = db_config['postgresql']
                connection_string = (
                    f"postgresql://{pg_config['user']}:{pg_config['password']}"
                    f"@{pg_config['host']}:{pg_config['port']}/{pg_config['database']}"
                )
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
        else:
            # Default SQLite
            db_path = './data/renewable_energy.db'
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            connection_string = f'sqlite:///{db_path}'

        return create_engine(connection_string, echo=False)

    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(self.engine)
        print("Database tables created successfully!")

    def get_session(self):
        """Get database session"""
        return self.SessionLocal()

    def close(self):
        """Close database connection"""
        self.engine.dispose()
