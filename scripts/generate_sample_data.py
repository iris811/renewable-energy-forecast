"""
Generate sample data for renewable energy forecasting
Creates synthetic solar/wind power generation data based on weather patterns
"""
import sys
import os
import numpy as np
from datetime import datetime, timedelta
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.database import DatabaseManager, WeatherData, PowerGeneration
from src.utils.logger import log


class SampleDataGenerator:
    """
    Generate realistic sample data for renewable energy forecasting
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def generate_weather_data(
        self,
        location_name: str,
        latitude: float,
        longitude: float,
        start_date: datetime,
        num_days: int = 365
    ):
        """
        Generate synthetic weather data

        Args:
            location_name: Location name
            latitude: Latitude
            longitude: Longitude
            start_date: Start date
            num_days: Number of days to generate
        """
        log.info(f"Generating weather data for {location_name}...")

        session = self.db_manager.get_session()
        count = 0

        try:
            for day in range(num_days):
                current_date = start_date + timedelta(days=day)

                # Generate 24 hourly records per day
                for hour in range(24):
                    timestamp = current_date + timedelta(hours=hour)

                    # Day of year for seasonal patterns
                    day_of_year = timestamp.timetuple().tm_yday

                    # Seasonal temperature variation (Seoul climate)
                    base_temp = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                    daily_variation = 8 * np.sin(2 * np.pi * hour / 24 - np.pi / 2)
                    temperature = base_temp + daily_variation + np.random.normal(0, 2)

                    # Humidity (inversely correlated with temperature)
                    humidity = 70 - 0.5 * (temperature - 15) + np.random.normal(0, 10)
                    humidity = np.clip(humidity, 20, 95)

                    # Pressure (realistic range)
                    pressure = 1013 + np.random.normal(0, 5)

                    # Wind speed (higher in winter)
                    base_wind = 3 + 2 * np.sin(2 * np.pi * (day_of_year - 270) / 365)
                    wind_speed = max(0, base_wind + np.random.normal(0, 1.5))

                    # Wind direction (random but biased)
                    wind_direction = np.random.uniform(0, 360)

                    # Cloud cover (affects solar generation)
                    cloud_cover = np.clip(np.random.beta(2, 5) * 100, 0, 100)

                    # Precipitation (rare events)
                    precipitation = np.random.exponential(0.5) if np.random.random() < 0.1 else 0

                    # Solar irradiance (depends on time and cloud cover)
                    solar_elevation = max(0, 90 * np.sin(2 * np.pi * (hour - 6) / 12))
                    base_irradiance = 1000 * np.sin(np.radians(solar_elevation))
                    ghi = max(0, base_irradiance * (1 - cloud_cover / 150) * (1 + 0.1 * np.sin(2 * np.pi * day_of_year / 365)))

                    # DNI and DHI (simplified)
                    dni = ghi * 0.8 if cloud_cover < 30 else ghi * 0.3
                    dhi = ghi - dni * np.sin(np.radians(solar_elevation))

                    # Create weather record
                    weather = WeatherData(
                        timestamp=timestamp,
                        location_name=location_name,
                        latitude=latitude,
                        longitude=longitude,
                        temperature=temperature,
                        humidity=humidity,
                        pressure=pressure,
                        wind_speed=wind_speed,
                        wind_direction=wind_direction,
                        cloud_cover=cloud_cover,
                        precipitation=precipitation,
                        solar_irradiance=ghi,
                        ghi=ghi,
                        dni=dni,
                        dhi=dhi,
                        source='simulated'
                    )

                    session.add(weather)
                    count += 1

                # Commit every day
                if (day + 1) % 10 == 0:
                    session.commit()
                    log.info(f"Generated {count} weather records...")

            session.commit()
            log.info(f"✓ Generated {count} weather records for {location_name}")

        except Exception as e:
            session.rollback()
            log.error(f"Error generating weather data: {e}")
            raise
        finally:
            session.close()

    def generate_power_data(
        self,
        location_name: str,
        energy_type: str,
        capacity: float,
        start_date: datetime,
        num_days: int = 365
    ):
        """
        Generate synthetic power generation data

        Args:
            location_name: Location name
            energy_type: 'solar' or 'wind'
            capacity: Installed capacity (kW)
            start_date: Start date
            num_days: Number of days to generate
        """
        log.info(f"Generating {energy_type} power data for {location_name}...")

        session = self.db_manager.get_session()
        count = 0

        try:
            # Get weather data for this location
            weather_query = session.query(WeatherData).filter(
                WeatherData.location_name == location_name,
                WeatherData.timestamp >= start_date,
                WeatherData.timestamp < start_date + timedelta(days=num_days)
            ).order_by(WeatherData.timestamp)

            weather_records = weather_query.all()

            if not weather_records:
                log.error("No weather data found! Generate weather data first.")
                return

            for weather in weather_records:
                if energy_type == 'solar':
                    power_output = self._calculate_solar_power(weather, capacity)
                elif energy_type == 'wind':
                    power_output = self._calculate_wind_power(weather, capacity)
                else:
                    power_output = 0

                # Add realistic noise
                power_output *= (1 + np.random.normal(0, 0.05))
                power_output = max(0, min(power_output, capacity))

                # Capacity factor
                capacity_factor = power_output / capacity if capacity > 0 else 0

                power = PowerGeneration(
                    timestamp=weather.timestamp,
                    location_name=location_name,
                    power_output=power_output,
                    energy_type=energy_type,
                    capacity=capacity,
                    capacity_factor=capacity_factor,
                    is_predicted=False
                )

                session.add(power)
                count += 1

                if count % 100 == 0:
                    session.commit()

            session.commit()
            log.info(f"✓ Generated {count} power generation records for {location_name}")

        except Exception as e:
            session.rollback()
            log.error(f"Error generating power data: {e}")
            raise
        finally:
            session.close()

    def _calculate_solar_power(self, weather: WeatherData, capacity: float) -> float:
        """
        Calculate solar power output based on weather

        Args:
            weather: Weather data
            capacity: Installed capacity

        Returns:
            Power output (kW)
        """
        if weather.ghi is None or weather.ghi <= 0:
            return 0

        # Base generation from GHI
        # Standard Test Conditions: 1000 W/m² produces rated capacity
        base_power = (weather.ghi / 1000) * capacity

        # Temperature derating (panels lose efficiency in heat)
        # Optimal temperature is 25°C, lose 0.4% per degree above
        temp_factor = 1 - 0.004 * (weather.temperature - 25) if weather.temperature else 1
        temp_factor = max(0.7, min(1.2, temp_factor))

        # Cloud cover effect (already in GHI, but add variance)
        cloud_factor = 1 - (weather.cloud_cover / 200) if weather.cloud_cover else 1

        # System efficiency (inverter, wiring losses, etc.)
        system_efficiency = 0.85

        power = base_power * temp_factor * cloud_factor * system_efficiency

        return max(0, power)

    def _calculate_wind_power(self, weather: WeatherData, capacity: float) -> float:
        """
        Calculate wind power output based on weather

        Args:
            weather: Weather data
            capacity: Installed capacity

        Returns:
            Power output (kW)
        """
        if weather.wind_speed is None:
            return 0

        v = weather.wind_speed

        # Typical wind turbine power curve
        # Cut-in: 3 m/s, Rated: 12 m/s, Cut-out: 25 m/s

        if v < 3:  # Below cut-in
            return 0
        elif v >= 25:  # Above cut-out (safety)
            return 0
        elif v >= 12:  # Rated wind speed
            return capacity
        else:  # Between cut-in and rated (cubic relationship)
            # Power proportional to v³ in this range
            power_fraction = ((v - 3) / (12 - 3)) ** 3
            return capacity * power_fraction


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate sample data')
    parser.add_argument('--location', type=str, default='Seoul Solar Farm', help='Location name')
    parser.add_argument('--latitude', type=float, default=37.5665, help='Latitude')
    parser.add_argument('--longitude', type=float, default=126.9780, help='Longitude')
    parser.add_argument('--energy-type', type=str, default='solar', choices=['solar', 'wind'], help='Energy type')
    parser.add_argument('--capacity', type=float, default=1000, help='Installed capacity (kW)')
    parser.add_argument('--days', type=int, default=365, help='Number of days to generate')
    parser.add_argument('--start-date', type=str, default=None, help='Start date (YYYY-MM-DD)')

    args = parser.parse_args()

    log.info("=" * 80)
    log.info("SAMPLE DATA GENERATION")
    log.info("=" * 80)

    # Parse start date
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_date = datetime.now() - timedelta(days=args.days)

    log.info(f"Location: {args.location}")
    log.info(f"Energy Type: {args.energy_type}")
    log.info(f"Capacity: {args.capacity} kW")
    log.info(f"Period: {start_date.date()} to {(start_date + timedelta(days=args.days)).date()}")
    log.info(f"Duration: {args.days} days\n")

    # Initialize database
    db_manager = DatabaseManager()
    db_manager.create_tables()

    # Generate data
    generator = SampleDataGenerator(db_manager)

    # Step 1: Generate weather data
    generator.generate_weather_data(
        location_name=args.location,
        latitude=args.latitude,
        longitude=args.longitude,
        start_date=start_date,
        num_days=args.days
    )

    # Step 2: Generate power generation data
    generator.generate_power_data(
        location_name=args.location,
        energy_type=args.energy_type,
        capacity=args.capacity,
        start_date=start_date,
        num_days=args.days
    )

    log.info("\n" + "=" * 80)
    log.info("✓ Sample data generation completed successfully!")
    log.info("=" * 80)

    # Show summary
    session = db_manager.get_session()
    weather_count = session.query(WeatherData).count()
    power_count = session.query(PowerGeneration).count()
    session.close()

    log.info(f"\nDatabase Summary:")
    log.info(f"  Weather records: {weather_count}")
    log.info(f"  Power records: {power_count}")
    log.info(f"  Ready for training!\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
