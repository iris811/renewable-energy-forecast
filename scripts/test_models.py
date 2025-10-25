"""
Test script to verify models work correctly
"""
import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import (
    create_model,
    print_model_summary,
    count_parameters,
    get_device,
    compare_models
)
from src.utils.logger import log


def test_model(model_type: str, input_dim: int = 50, output_dim: int = 24, sequence_length: int = 168):
    """
    Test a specific model

    Args:
        model_type: Type of model to test
        input_dim: Number of input features
        output_dim: Prediction horizon
        sequence_length: Input sequence length
    """
    log.info(f"\n{'='*60}")
    log.info(f"Testing {model_type} model")
    log.info(f"{'='*60}")

    try:
        # Create model
        model = create_model(
            model_type=model_type,
            input_dim=input_dim,
            output_dim=output_dim,
            sequence_length=sequence_length
        )

        # Get device
        device = get_device()
        model = model.to(device)

        # Print summary
        print_model_summary(model)

        # Test forward pass
        batch_size = 32
        x = torch.randn(batch_size, sequence_length, input_dim).to(device)

        log.info("Testing forward pass...")
        model.eval()
        with torch.no_grad():
            output = model(x)

        log.info(f"Input shape: {x.shape}")
        log.info(f"Output shape: {output.shape}")

        # Verify output shape
        expected_shape = (batch_size, output_dim)
        if output.shape == expected_shape:
            log.info(f"✓ Output shape is correct: {output.shape}")
        else:
            log.error(f"✗ Output shape mismatch! Expected {expected_shape}, got {output.shape}")
            return False

        log.info(f"✓ {model_type} model test passed!")
        return True

    except Exception as e:
        log.error(f"✗ {model_type} model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_models():
    """Test all available models"""
    log.info("\n" + "="*80)
    log.info("TESTING ALL MODELS")
    log.info("="*80 + "\n")

    # Define test parameters
    input_dim = 50
    output_dim = 24
    sequence_length = 168

    # Models to test
    model_types = [
        'lstm',
        'lstm_attention',
        'transformer',
        'timeseries',
    ]

    results = {}
    for model_type in model_types:
        success = test_model(model_type, input_dim, output_dim, sequence_length)
        results[model_type] = success

    # Print summary
    log.info("\n" + "="*80)
    log.info("TEST SUMMARY")
    log.info("="*80)

    for model_type, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        log.info(f"{model_type:<20} {status}")

    log.info("="*80 + "\n")

    # Overall result
    all_passed = all(results.values())
    if all_passed:
        log.info("✓ All tests passed!")
        return 0
    else:
        log.error("✗ Some tests failed!")
        return 1


def compare_all_models():
    """Compare all models"""
    log.info("\n" + "="*80)
    log.info("COMPARING MODELS")
    log.info("="*80 + "\n")

    input_dim = 50
    output_dim = 24
    sequence_length = 168

    models = {}

    # Create models
    model_types = ['lstm', 'lstm_attention', 'transformer', 'timeseries']

    for model_type in model_types:
        try:
            model = create_model(
                model_type=model_type,
                input_dim=input_dim,
                output_dim=output_dim,
                sequence_length=sequence_length
            )
            models[model_type] = model
        except Exception as e:
            log.warning(f"Could not create {model_type}: {e}")

    # Compare
    if models:
        compare_models(models)


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Test models')
    parser.add_argument(
        '--model',
        type=str,
        default='all',
        help='Model type to test (all, lstm, transformer, etc.)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all models'
    )

    args = parser.parse_args()

    if args.compare:
        compare_all_models()
        return 0

    if args.model == 'all':
        return test_all_models()
    else:
        success = test_model(args.model)
        return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
