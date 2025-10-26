"""
Evaluate trained model on test set
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import yaml

from src.preprocessing.data_pipeline import DataPipeline
from src.models.model_utils import load_model
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.visualizer import PredictionVisualizer
from src.utils.logger import get_logger
from src.preprocessing.scaler import load_scaler

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model")

    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--location', type=str, default='Seoul Solar Farm',
                       help='Location name')
    parser.add_argument('--energy-type', type=str, default='solar',
                       choices=['solar', 'wind'],
                       help='Energy type')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (-1 for CPU)')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')

    return parser.parse_args()


def main():
    """Main evaluation function"""

    args = parse_args()

    print("=" * 80)
    print("RENEWABLE ENERGY FORECASTING - MODEL EVALUATION")
    print("=" * 80)
    print()

    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'
    logger.info(f"Using device: {device}")

    # Load config
    config_path = Path('configs/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Step 1: Prepare data
    print("\nStep 1: Loading test data...")
    print("-" * 80)

    pipeline = DataPipeline(
        location=args.location,
        energy_type=args.energy_type,
        sequence_length=config['model']['sequence_length'],
        prediction_horizon=config['model']['prediction_horizon']
    )

    try:
        train_loader, val_loader, test_loader = pipeline.prepare_data(
            batch_size=args.batch_size,
            test_size=0.2,
            val_size=0.1
        )
        logger.info(f"✓ Test data loaded: {len(test_loader)} batches")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Step 2: Load model
    print("\nStep 2: Loading model...")
    print("-" * 80)

    try:
        model = load_model(args.model_path, device=device)
        logger.info(f"✓ Model loaded from {args.model_path}")

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Model parameters: {num_params:,}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Step 3: Load scaler
    print("\nStep 3: Loading scaler...")
    print("-" * 80)

    try:
        scaler_name = f"{args.location}_{args.energy_type}_target_scaler.pkl"
        scaler_path = Path('models/scalers') / scaler_name
        target_scaler = load_scaler(str(scaler_path))
        logger.info(f"✓ Scaler loaded from {scaler_path}")
    except Exception as e:
        logger.warning(f"Could not load scaler: {e}")
        target_scaler = None

    # Step 4: Evaluate model
    print("\nStep 4: Evaluating model...")
    print("-" * 80)

    evaluator = ModelEvaluator(model, device=device)

    try:
        metrics = evaluator.evaluate(
            test_loader,
            scaler=target_scaler,
            save_predictions=True,
            output_dir=args.output_dir
        )

        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"\nOverall Performance:")
        print(f"  RMSE:  {metrics['rmse']:.6f}")
        print(f"  MAE:   {metrics['mae']:.6f}")
        print(f"  MAPE:  {metrics['mape']:.2f}%")
        print(f"  R²:    {metrics['r2']:.6f}")

        if 'error_stats' in metrics:
            print(f"\nError Statistics:")
            stats = metrics['error_stats']
            print(f"  Mean:   {stats['mean']:.6f}")
            print(f"  Std:    {stats['std']:.6f}")
            print(f"  Median: {stats['q50']:.6f}")

        print("\n" + "=" * 80)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Create visualizations
    if args.visualize:
        print("\nStep 5: Creating visualizations...")
        print("-" * 80)

        visualizer = PredictionVisualizer(
            output_dir=f"{args.output_dir}/plots"
        )

        try:
            # Main dashboard
            visualizer.create_evaluation_dashboard(
                evaluator.predictions,
                evaluator.actuals,
                metrics,
                save_name='evaluation_dashboard.png'
            )
            logger.info("✓ Created evaluation dashboard")

            # Predictions plot
            visualizer.plot_predictions(
                evaluator.predictions,
                evaluator.actuals,
                title=f"{args.location} - {args.energy_type.title()}",
                save_name='predictions_vs_actuals.png'
            )
            logger.info("✓ Created predictions plot")

            # Error analysis
            visualizer.plot_error_analysis(
                evaluator.predictions,
                evaluator.actuals,
                save_name='error_analysis.png'
            )
            logger.info("✓ Created error analysis plot")

            # Horizon performance (if multi-step)
            if 'horizon_metrics' in metrics:
                visualizer.plot_horizon_performance(
                    metrics['horizon_metrics'],
                    save_name='horizon_performance.png'
                )
                logger.info("✓ Created horizon performance plot")

        except Exception as e:
            logger.warning(f"Visualization failed: {e}")

    print("\n" + "=" * 80)
    print("✓ Evaluation completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
