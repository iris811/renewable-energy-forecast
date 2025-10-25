"""
Data loader for fetching and preparing data from database
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from sqlalchemy import and_
from ..utils.database import DatabaseManager, WeatherData, PowerGeneration
from ..utils.logger import log


class DataLoader:
    """
    Loads data from database and converts to pandas DataFrames
    """

    def __init__(self, config_path='configs/config.yaml'):
        """
        Initialize data loader

        Args:
            config_path: Path to configuration file
        """
        self.db_manager = DatabaseManager(config_path)

    def load_weather_data(
        self,
        location_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load weather data from database

        Args:
            location_name: Filter by location name
            start_date: Start date for data
            end_date: End date for data
            source: Filter by data source (e.g., 'openweathermap', 'nasa_power')

        Returns:
            DataFrame with weather data
        """
        session = self.db_manager.get_session()

        try:
            query = session.query(WeatherData)

            # Apply filters
            if location_name:
                query = query.filter(WeatherData.location_name == location_name)
            if start_date:
                query = query.filter(WeatherData.timestamp >= start_date)
            if end_date:
                query = query.filter(WeatherData.timestamp <= end_date)
            if source:
                query = query.filter(WeatherData.source == source)

            # Order by timestamp
            query = query.order_by(WeatherData.timestamp)

            # Fetch data
            results = query.all()

            if not results:
                log.warning(f"No weather data found for filters: location={location_name}, source={source}")
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for row in results:
                data.append({
                    'timestamp': row.timestamp,
                    'location_name': row.location_name,
                    'latitude': row.latitude,
                    'longitude': row.longitude,
                    'temperature': row.temperature,
                    'humidity': row.humidity,
                    'pressure': row.pressure,
                    'wind_speed': row.wind_speed,
                    'wind_direction': row.wind_direction,
                    'cloud_cover': row.cloud_cover,
                    'precipitation': row.precipitation,
                    'solar_irradiance': row.solar_irradiance,
                    'ghi': row.ghi,
                    'dni': row.dni,
                    'dhi': row.dhi,
                    'source': row.source
                })

            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            log.info(f"Loaded {len(df)} weather records")
            return df

        except Exception as e:
            log.error(f"Error loading weather data: {e}")
            return pd.DataFrame()
        finally:
            session.close()

    def load_power_generation(
        self,
        location_name: Optional[str] = None,
        energy_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_predicted: bool = False
    ) -> pd.DataFrame:
        """
        Load power generation data from database

        Args:
            location_name: Filter by location name
            energy_type: Filter by energy type ('solar' or 'wind')
            start_date: Start date for data
            end_date: End date for data
            include_predicted: Include predicted values

        Returns:
            DataFrame with power generation data
        """
        session = self.db_manager.get_session()

        try:
            query = session.query(PowerGeneration)

            # Apply filters
            if location_name:
                query = query.filter(PowerGeneration.location_name == location_name)
            if energy_type:
                query = query.filter(PowerGeneration.energy_type == energy_type)
            if start_date:
                query = query.filter(PowerGeneration.timestamp >= start_date)
            if end_date:
                query = query.filter(PowerGeneration.timestamp <= end_date)
            if not include_predicted:
                query = query.filter(PowerGeneration.is_predicted == False)

            # Order by timestamp
            query = query.order_by(PowerGeneration.timestamp)

            # Fetch data
            results = query.all()

            if not results:
                log.warning(f"No power generation data found for filters: location={location_name}, type={energy_type}")
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for row in results:
                data.append({
                    'timestamp': row.timestamp,
                    'location_name': row.location_name,
                    'power_output': row.power_output,
                    'energy_type': row.energy_type,
                    'capacity': row.capacity,
                    'capacity_factor': row.capacity_factor,
                    'is_predicted': row.is_predicted
                })

            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            log.info(f"Loaded {len(df)} power generation records")
            return df

        except Exception as e:
            log.error(f"Error loading power generation data: {e}")
            return pd.DataFrame()
        finally:
            session.close()

    def merge_weather_and_power(
        self,
        location_name: str,
        energy_type: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Merge weather and power generation data for training

        Args:
            location_name: Location name
            energy_type: Energy type ('solar' or 'wind')
            start_date: Start date for data
            end_date: End date for data

        Returns:
            Merged DataFrame with both weather and power data
        """
        # Load weather data
        weather_df = self.load_weather_data(
            location_name=location_name,
            start_date=start_date,
            end_date=end_date
        )

        # Load power generation data
        power_df = self.load_power_generation(
            location_name=location_name,
            energy_type=energy_type,
            start_date=start_date,
            end_date=end_date,
            include_predicted=False
        )

        if weather_df.empty or power_df.empty:
            log.warning("Cannot merge: weather or power data is empty")
            return pd.DataFrame()

        # Merge on timestamp
        merged_df = weather_df.join(power_df[['power_output', 'energy_type', 'capacity', 'capacity_factor']], how='inner')

        log.info(f"Merged data: {len(merged_df)} records")
        return merged_df

    def get_data_summary(self, location_name: Optional[str] = None) -> dict:
        """
        Get summary statistics of available data

        Args:
            location_name: Filter by location name

        Returns:
            Dictionary with data summary
        """
        session = self.db_manager.get_session()

        try:
            summary = {}

            # Weather data summary
            weather_query = session.query(WeatherData)
            if location_name:
                weather_query = weather_query.filter(WeatherData.location_name == location_name)

            weather_count = weather_query.count()
            if weather_count > 0:
                first_weather = weather_query.order_by(WeatherData.timestamp).first()
                last_weather = weather_query.order_by(WeatherData.timestamp.desc()).first()
                summary['weather'] = {
                    'count': weather_count,
                    'first_date': first_weather.timestamp,
                    'last_date': last_weather.timestamp,
                    'locations': [loc[0] for loc in session.query(WeatherData.location_name).distinct().all()]
                }

            # Power generation summary
            power_query = session.query(PowerGeneration)
            if location_name:
                power_query = power_query.filter(PowerGeneration.location_name == location_name)

            power_count = power_query.count()
            if power_count > 0:
                first_power = power_query.order_by(PowerGeneration.timestamp).first()
                last_power = power_query.order_by(PowerGeneration.timestamp.desc()).first()
                summary['power'] = {
                    'count': power_count,
                    'first_date': first_power.timestamp,
                    'last_date': last_power.timestamp
                }

            return summary

        except Exception as e:
            log.error(f"Error getting data summary: {e}")
            return {}
        finally:
            session.close()

    def get_locations(self) -> List[str]:
        """
        Get list of all locations with data

        Returns:
            List of location names
        """
        session = self.db_manager.get_session()
        try:
            locations = [loc[0] for loc in session.query(WeatherData.location_name).distinct().all()]
            return locations
        finally:
            session.close()
