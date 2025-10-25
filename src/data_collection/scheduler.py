"""
Data collection scheduler for automatic periodic data collection
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import yaml
import os
from ..utils.logger import log
from .weather_collector import WeatherCollector
from .nasa_power_collector import NASAPowerCollector


class DataCollectionScheduler:
    """
    Manages scheduled data collection tasks
    """

    def __init__(self, config_path='configs/config.yaml'):
        """
        Initialize scheduler

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.scheduler = BackgroundScheduler()
        self.weather_collector = None
        self.nasa_collector = None

        # Initialize collectors
        self._init_collectors()

    def _load_config(self, config_path):
        """Load configuration file"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}

    def _init_collectors(self):
        """Initialize data collectors"""
        sources = self.config.get('data', {}).get('collection', {}).get('sources', [])

        for source in sources:
            if source.get('name') == 'weather_api' and source.get('enabled'):
                self.weather_collector = WeatherCollector()
                log.info("Weather collector initialized")

            elif source.get('name') == 'nasa_power' and source.get('enabled'):
                self.nasa_collector = NASAPowerCollector()
                log.info("NASA POWER collector initialized")

    def collect_weather_data(self):
        """Job: Collect weather data from all locations"""
        try:
            log.info("Starting scheduled weather data collection...")
            if self.weather_collector:
                self.weather_collector.collect_all_locations()
                log.info("Weather data collection completed successfully")
            else:
                log.warning("Weather collector not initialized")
        except Exception as e:
            log.error(f"Error in weather data collection: {e}")

    def collect_nasa_data(self):
        """Job: Collect NASA POWER data from all locations"""
        try:
            log.info("Starting scheduled NASA POWER data collection...")
            if self.nasa_collector:
                # Collect last 7 days of data
                self.nasa_collector.collect_all_locations(days_back=7)
                log.info("NASA POWER data collection completed successfully")
            else:
                log.warning("NASA POWER collector not initialized")
        except Exception as e:
            log.error(f"Error in NASA POWER data collection: {e}")

    def start(self):
        """Start the scheduler"""
        sources = self.config.get('data', {}).get('collection', {}).get('sources', [])

        # Schedule weather data collection
        weather_source = next((s for s in sources if s.get('name') == 'weather_api'), None)
        if weather_source and weather_source.get('enabled'):
            interval = weather_source.get('update_interval', 3600)  # Default 1 hour
            self.scheduler.add_job(
                func=self.collect_weather_data,
                trigger=IntervalTrigger(seconds=interval),
                id='weather_collection',
                name='Weather Data Collection',
                replace_existing=True
            )
            log.info(f"Scheduled weather data collection every {interval} seconds")

        # Schedule NASA POWER data collection
        nasa_source = next((s for s in sources if s.get('name') == 'nasa_power'), None)
        if nasa_source and nasa_source.get('enabled'):
            interval = nasa_source.get('update_interval', 86400)  # Default 1 day
            self.scheduler.add_job(
                func=self.collect_nasa_data,
                trigger=IntervalTrigger(seconds=interval),
                id='nasa_collection',
                name='NASA POWER Data Collection',
                replace_existing=True
            )
            log.info(f"Scheduled NASA POWER data collection every {interval} seconds")

        # Start scheduler
        self.scheduler.start()
        log.info("Data collection scheduler started")

        # Run initial collection immediately
        log.info("Running initial data collection...")
        if self.weather_collector:
            self.collect_weather_data()
        if self.nasa_collector:
            self.collect_nasa_data()

    def stop(self):
        """Stop the scheduler"""
        self.scheduler.shutdown()
        log.info("Data collection scheduler stopped")

    def get_jobs(self):
        """Get list of scheduled jobs"""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time
            })
        return jobs

    def pause_job(self, job_id: str):
        """
        Pause a specific job

        Args:
            job_id: Job identifier
        """
        self.scheduler.pause_job(job_id)
        log.info(f"Paused job: {job_id}")

    def resume_job(self, job_id: str):
        """
        Resume a paused job

        Args:
            job_id: Job identifier
        """
        self.scheduler.resume_job(job_id)
        log.info(f"Resumed job: {job_id}")


def run_scheduler():
    """Run the data collection scheduler (blocking)"""
    scheduler = DataCollectionScheduler()
    scheduler.start()

    try:
        # Keep the scheduler running
        import time
        log.info("Scheduler is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Stopping scheduler...")
        scheduler.stop()


if __name__ == '__main__':
    run_scheduler()
