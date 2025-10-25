"""
Data preprocessing script
Demonstrates the complete preprocessing pipeline
"""
import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import DataPipeline
from src.utils.logger import log


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Preprocess data for renewable energy forecasting')
    parser.add_argument(
        '--location',
        type=str,
        default='Seoul Solar Farm',
        help='Location name'
    )
    parser.add_argument(
        '--energy-type',
        type=str,
        choices=['solar', 'wind'],
        default='solar',
        help='Energy type'
    )
    parser.add_argument(
        '--latitude',
        type=float,
        default=37.5665,
        help='Location latitude'
    )
    parser.add_argument(
        '--longitude',
        type=float,
        default=126.9780,
        help='Location longitude'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=168,
        help='Sequence length in hours (default: 168 = 1 week)'
    )
    parser.add_argument(
        '--prediction-horizon',
        type=int,
        default=24,
        help='Prediction horizon in hours (default: 24)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--save-processed',
        action='store_true',
        help='Save processed data to CSV'
    )

    args = parser.parse_args()

    # Create pipeline
    log.info(f"Creating preprocessing pipeline for {args.location} ({args.energy_type})")
    pipeline = DataPipeline(
        location_name=args.location,
        energy_type=args.energy_type,
        latitude=args.latitude,
        longitude=args.longitude,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon
    )

    # Run pipeline
    try:
        result = pipeline.run_pipeline(batch_size=args.batch_size)

        if not result:
            log.error("Pipeline failed - no data returned")
            return 1

        # Print summary
        log.info("\n" + "=" * 60)
        log.info("PREPROCESSING SUMMARY")
        log.info("=" * 60)
        log.info(f"Location: {args.location}")
        log.info(f"Energy Type: {args.energy_type}")
        log.info(f"Number of features: {result['n_features']}")
        log.info(f"Sequence length: {result['sequence_length']} hours")
        log.info(f"Prediction horizon: {result['prediction_horizon']} hours")
        log.info(f"Train batches: {len(result['train_loader'])}")
        log.info(f"Validation batches: {len(result['val_loader'])}")
        log.info(f"Test batches: {len(result['test_loader'])}")
        log.info("=" * 60)

        # Save processed data if requested
        if args.save_processed and pipeline.processed_data is not None:
            output_dir = 'data/processed'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir,
                f'{args.location.replace(" ", "_")}_{args.energy_type}_processed.csv'
            )
            pipeline.processed_data.to_csv(output_path)
            log.info(f"Saved processed data to {output_path}")

        # Print data quality report
        summary = pipeline.get_data_summary()
        if summary:
            log.info("\nDATA QUALITY REPORT")
            log.info(f"Total records: {summary.get('total_records', 'N/A')}")
            log.info(f"Date range: {summary.get('date_range', {}).get('start')} to {summary.get('date_range', {}).get('end')}")
            log.info(f"Duplicates: {summary.get('duplicates', 'N/A')}")

        log.info("\n✓ Preprocessing completed successfully!")
        return 0

    except Exception as e:
        log.error(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
