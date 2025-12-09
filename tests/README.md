# BQNN Unit Tests

This directory contains comprehensive unit tests for the BQNN benchmark project.

## Test Coverage

The test suite covers the following components:

### 1. `test_model.py` - BQNNModel Tests
Tests for the quantum-classical hybrid neural network model:
- **Forward pass correctness**: Verifies that `BQNNModel.forward()` correctly processes input, invokes the quantum circuit, and produces logits of the expected shape
- **Quantum circuit invocation**: Confirms the quantum layer is properly invoked during forward pass
- **Batch processing**: Ensures `quantum_forward()` correctly handles batched inputs
- **Gradient flow**: Tests that gradients propagate through all model parameters
- **Quantumness parameter**: Validates setting and behavior of the quantumness parameter `a`
- **Noise configuration**: Tests noise injection functionality
- **Circuit information**: Verifies circuit metadata reporting

### 2. `test_ste.py` - Straight-Through Estimator Tests
Tests for gradient estimators used in binarized networks:
- **StraightThroughEstimator forward**: Verifies correct application of `sign()` function
- **StraightThroughEstimator backward**: Confirms identity gradient propagation
- **ClippedSTE forward**: Tests correct sign function application
- **ClippedSTE backward**: Validates gradient clipping to [-1, 1] region
- **Gradient flow**: Ensures STE allows gradients to flow through networks
- **Comparison tests**: Compares standard vs clipped STE behavior

### 3. `test_training.py` - Trainer Class Tests
Tests for the training utilities:
- **Initialization**: Validates proper Trainer setup
- **Single training step**: Tests that `train_epoch()` performs optimizer updates correctly
- **Gradient clipping**: Confirms gradient clipping is applied
- **Gradient tracking**: Tests optional gradient statistics collection
- **Early stopping**: Validates early stopping patience counter
- **Learning rate scheduling**: Tests StepLR scheduler integration
- **Optimizer persistence**: Ensures optimizer state persists across epochs
- **History tracking**: Verifies training history is correctly maintained

### 4. `test_data.py` - Data Loading Tests
Tests for dataset generation and loading:
- **Synthetic dataset generation**: Tests `get_synthetic_dataset()` with various parameters
- **Seed reproducibility**: Ensures same seed produces identical data
- **Batch size handling**: Validates correct batch size application
- **Train/test splitting**: Tests `get_train_test_loaders()` with proper seed separation
- **Output format validation**: Confirms correct DataLoader format
- **Parameter propagation**: Tests passing additional kwargs to dataset functions

### 5. `test_metrics.py` - Evaluation Metrics Tests
Tests for classification performance metrics:
- **Accuracy computation**: Tests `evaluate_accuracy()` with perfect, zero, and partial accuracy
- **F1 score calculation**: Tests `compute_f1_score()` for binary and multiclass scenarios
- **Confusion matrix**: Tests `compute_confusion_matrix()` generation
- **Edge cases**: Validates handling of empty data, imbalanced classes, and missing classes
- **Metrics consistency**: Integration tests ensuring all metrics agree

## Running the Tests

### Prerequisites
Ensure you have Python 3.10+ and the dev dependencies installed:

```bash
pip install -e ".[dev]"
```

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_model.py -v
pytest tests/test_ste.py -v
pytest tests/test_training.py -v
pytest tests/test_data.py -v
pytest tests/test_metrics.py -v
```

### Run Specific Test Class or Function
```bash
# Run a specific test class
pytest tests/test_model.py::TestBQNNModel -v

# Run a specific test function
pytest tests/test_model.py::TestBQNNModel::test_forward_basic -v
```

### Run with Coverage
```bash
pytest tests/ --cov=bqnn --cov-report=html
```

This generates an HTML coverage report in `htmlcov/`.

## Test Structure

Each test file follows a consistent structure:
- Tests are organized into classes for logical grouping
- Descriptive test names explain what is being tested
- Docstrings provide additional context
- Assertions include helpful error messages

## Writing New Tests

When adding new tests:
1. Follow the existing naming convention: `test_<feature>.py`
2. Group related tests in classes: `class TestFeatureName`
3. Use descriptive test names: `def test_specific_behavior(self)`
4. Include docstrings explaining the test purpose
5. Use clear assertion messages for failures

## Notes

- Tests use CPU device by default for faster execution
- Quantum circuit tests use PennyLane's `default.qubit` simulator
- Some tests may be probabilistic (e.g., random data generation); fixed seeds ensure reproducibility
- Tests are designed to be independent and can run in any order
