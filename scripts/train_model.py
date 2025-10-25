"""
Train renewable energy forecasting model
"""
import sys
import os
import argparse
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import DataPipeline
from src.models import create_model, get_device
from src.training import Trainer, get_loss_function
from src.utils.logger import log
import yaml


def load_config(config_path='configs/config.yaml'):
    """Load configuration file"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train renewable energy forecasting model')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file path')
    parser.add_argument('--model-type', type=str, default='lstm', help='Model type')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--location', type=str, default='Seoul Solar Farm', help='Location name')
    parser.add_argument('--energy-type', type=str, default='solar', help='Energy type')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    log.info("=" * 80)
    log.info("RENEWABLE ENERGY FORECASTING - MODEL TRAINING")
    log.info("=" * 80)

    # Get device
    device = get_device(args.gpu)

    # Model configuration
    model_config = config.get('model', {})
    model_type_config = model_config.get(args.model_type, {})

    # Training configuration
    training_config = config.get('training', {})

    # Data configuration
    data_config = config.get('data', {})
    locations = data_config.get('collection', {}).get('locations', [])

    # Find location
    location_info = None
    for loc in locations:
        if loc['name'] == args.location:
            location_info = loc
            break

    if not location_info:
        log.error(f"Location not found in config: {args.location}")
        log.info(f"Available locations: {[loc['name'] for loc in locations]}")
        return 1

    # Extract location info
    latitude = location_info['latitude']
    longitude = location_info['longitude']

    log.info(f"\nConfiguration:")
    log.info(f"  Model: {args.model_type}")
    log.info(f"  Location: {args.location}")
    log.info(f"  Energy Type: {args.energy_type}")
    log.info(f"  Batch Size: {args.batch_size}")
    log.info(f"  Epochs: {args.epochs}")
    log.info(f"  Device: {device}\n")

    # Step 1: Prepare data
    log.info("Step 1: Preparing data...")
    pipeline = DataPipeline(
        location_name=args.location,
        energy_type=args.energy_type,
        latitude=latitude,
        longitude=longitude,
        sequence_length=model_config.get('sequence_length', 168),
        prediction_horizon=model_config.get('prediction_horizon', 24)
    )

    result = pipeline.run_pipeline(batch_size=args.batch_size)

    if not result:
        log.error("Data pipeline failed!")
        return 1

    train_loader = result['train_loader']
    val_loader = result['val_loader']
    test_loader = result['test_loader']
    n_features = result['n_features']

    log.info(f"✓ Data prepared successfully")
    log.info(f"  Features: {n_features}")
    log.info(f"  Train batches: {len(train_loader)}")
    log.info(f"  Val batches: {len(val_loader)}")
    log.info(f"  Test batches: {len(test_loader)}\n")

    # Step 2: Create model
    log.info("Step 2: Creating model...")
    model = create_model(
        model_type=args.model_type,
        input_dim=n_features,
        output_dim=model_config.get('prediction_horizon', 24),
        sequence_length=model_config.get('sequence_length', 168),
        config=model_type_config
    )

    log.info(f"✓ Model created successfully")
    log.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Step 3: Setup training
    log.info("Step 3: Setting up training...")

    # Loss function
    criterion = get_loss_function('rmse')

    # Optimizer
    lr = args.lr if args.lr is not None else model_type_config.get('learning_rate', 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training config
    train_config = {
        'loss': 'rmse',
        'learning_rate': lr,
        'early_stopping': training_config.get('early_stopping', {}),
        'scheduler': training_config.get('scheduler', {}),
        'checkpoint': training_config.get('checkpoint', {}),
        'grad_clip': training_config.get('grad_clip', 0)
    }

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=train_config
    )

    log.info("✓ Trainer initialized\n")

    # Step 4: Train
    log.info("Step 4: Training model...")
    history = trainer.train(epochs=args.epochs, resume_from=args.resume)

    log.info("✓ Training completed\n")

    # Step 5: Evaluate
    log.info("Step 5: Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)

    log.info("✓ Evaluation completed\n")

    # Print summary
    log.info("=" * 80)
    log.info("TRAINING SUMMARY")
    log.info("=" * 80)
    log.info(f"Best Validation RMSE: {trainer.best_val_loss:.4f}")
    log.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
    log.info(f"Test MAE: {test_metrics['mae']:.4f}")
    log.info(f"Test MAPE: {test_metrics['mape']:.2f}%")
    log.info(f"Test R2: {test_metrics['r2']:.4f}")
    log.info("=" * 80)

    # Save final results
    results_file = os.path.join(trainer.checkpoint_dir, 'training_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Location: {args.location}\n")
        f.write(f"Energy Type: {args.energy_type}\n")
        f.write(f"Epochs Trained: {len(history['train_loss'])}\n")
        f.write(f"Best Val RMSE: {trainer.best_val_loss:.4f}\n")
        f.write(f"Test RMSE: {test_metrics['rmse']:.4f}\n")
        f.write(f"Test MAE: {test_metrics['mae']:.4f}\n")
        f.write(f"Test MAPE: {test_metrics['mape']:.2f}%\n")
        f.write(f"Test R2: {test_metrics['r2']:.4f}\n")

    log.info(f"\n✓ Results saved to {results_file}")
    log.info("\n✓ Training pipeline completed successfully!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
