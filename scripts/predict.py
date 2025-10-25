"""
Make predictions using trained model
"""
import sys
import os
import argparse
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference import Predictor
from src.preprocessing import DataLoader
from src.utils.logger import log
import matplotlib.pyplot as plt
import yaml


def load_config(config_path='configs/config.yaml'):
    """Load configuration file"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def plot_predictions(predictions: dict, output_path: str):
    """
    Plot predictions

    Args:
        predictions: Prediction dictionary
        output_path: Output file path
    """
    plt.figure(figsize=(12, 6))

    timestamps = predictions['timestamps']
    pred_values = predictions['predictions']

    plt.plot(timestamps, pred_values, 'b-', linewidth=2, label='Predictions')

    # If uncertainty available
    if 'lower_bound' in predictions and 'upper_bound' in predictions:
        plt.fill_between(
            timestamps,
            predictions['lower_bound'],
            predictions['upper_bound'],
            alpha=0.3,
            label='90% Confidence Interval'
        )

    plt.xlabel('Time')
    plt.ylabel('Power Output (kW)')
    plt.title('Renewable Energy Power Generation Forecast')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    log.info(f"Plot saved to {output_path}")
    plt.close()


def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description='Make predictions')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--scaler-dir', type=str, default='models/scalers', help='Scalers directory')
    parser.add_argument('--location', type=str, default='Seoul Solar Farm', help='Location name')
    parser.add_argument('--energy-type', type=str, default='solar', help='Energy type')
    parser.add_argument('--output', type=str, default='predictions/forecast.csv', help='Output file path')
    parser.add_argument('--plot', action='store_true', help='Generate plot')
    parser.add_argument('--uncertainty', action='store_true', help='Estimate uncertainty')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file')

    args = parser.parse_args()

    log.info("=" * 80)
    log.info("RENEWABLE ENERGY FORECASTING - PREDICTION")
    log.info("=" * 80)

    # Load config
    config = load_config(args.config)
    data_config = config.get('data', {})
    locations = data_config.get('collection', {}).get('locations', [])

    # Find location info
    location_info = None
    for loc in locations:
        if loc['name'] == args.location:
            location_info = loc
            break

    if not location_info:
        log.error(f"Location not found in config: {args.location}")
        return 1

    latitude = location_info['latitude']
    longitude = location_info['longitude']

    log.info(f"\nConfiguration:")
    log.info(f"  Location: {args.location}")
    log.info(f"  Energy Type: {args.energy_type}")
    log.info(f"  Checkpoint: {args.checkpoint}")
    log.info(f"  Uncertainty: {args.uncertainty}\n")

    # Step 1: Load predictor
    log.info("Step 1: Loading predictor...")
    try:
        predictor = Predictor.from_checkpoint(
            checkpoint_path=args.checkpoint,
            scaler_dir=args.scaler_dir,
            location_name=args.location,
            energy_type=args.energy_type,
            latitude=latitude,
            longitude=longitude
        )
        log.info("✓ Predictor loaded\n")
    except Exception as e:
        log.error(f"Failed to load predictor: {e}")
        return 1

    # Step 2: Load recent data
    log.info("Step 2: Loading recent data...")
    data_loader = DataLoader()

    # Get the most recent data (last 7 days + 7 days for features)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=14)

    weather_df = data_loader.load_weather_data(
        location_name=args.location,
        start_date=start_date,
        end_date=end_date
    )

    if weather_df.empty:
        log.error("No recent data found. Please run data collection first.")
        log.info("Hint: python scripts/generate_sample_data.py")
        return 1

    log.info(f"✓ Loaded {len(weather_df)} weather records")
    log.info(f"  Date range: {weather_df.index.min()} to {weather_df.index.max()}\n")

    # Step 3: Make prediction
    log.info("Step 3: Making prediction...")
    try:
        if args.uncertainty:
            log.info("Estimating uncertainty (this may take a moment)...")
            result = predictor.predict_with_uncertainty(weather_df, n_samples=100)
        else:
            result = predictor.predict(weather_df)

        log.info("✓ Prediction completed\n")
    except Exception as e:
        log.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 4: Display results
    log.info("=" * 80)
    log.info("PREDICTION RESULTS")
    log.info("=" * 80)
    log.info(f"Prediction made at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Input data: {result['input_start']} to {result['input_end']}")
    log.info(f"Forecast period: {result['prediction_start']} to {result['prediction_end']}")
    log.info(f"Forecast horizon: {result['prediction_horizon']} hours\n")

    # Show first few predictions
    log.info("First 6 hours forecast:")
    for i in range(min(6, len(result['predictions']))):
        timestamp = result['timestamps'][i]
        power = result['predictions'][i]

        if 'std' in result:
            std = result['std'][i]
            lower = result['lower_bound'][i]
            upper = result['upper_bound'][i]
            log.info(f"  {timestamp}: {power:.2f} kW (±{std:.2f}, 90% CI: [{lower:.2f}, {upper:.2f}])")
        else:
            log.info(f"  {timestamp}: {power:.2f} kW")

    if len(result['predictions']) > 6:
        log.info(f"  ... (showing 6/{len(result['predictions'])} hours)")

    # Statistics
    log.info(f"\nForecast statistics:")
    log.info(f"  Mean: {result['predictions'].mean():.2f} kW")
    log.info(f"  Max: {result['predictions'].max():.2f} kW")
    log.info(f"  Min: {result['predictions'].min():.2f} kW")
    log.info("=" * 80)

    # Step 5: Save results
    log.info("\nStep 5: Saving results...")
    predictor.save_predictions(result, args.output)

    # Generate plot if requested
    if args.plot:
        plot_path = args.output.replace('.csv', '.png')
        plot_predictions(result, plot_path)

    log.info("\n✓ Prediction completed successfully!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
