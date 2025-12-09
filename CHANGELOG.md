# BQNN Benchmark Changelog

## [0.2.1] - 2025-12-09

### Critical Bug Fixes (Quantum Layer)

#### 1. PauliZ Measurement → Zero Gradient
**Severity: CRITICAL**

The quantum circuit measured PauliZ after RZ rotations:
```
|0⟩ → RX(input) → RZ(θ) → measure Z
```
Since RZ only adds phase (invisible to Z measurement):
- ⟨Z⟩ = cos(input) — **no θ dependence**
- ∂⟨Z⟩/∂θ = 0 exactly

**Fix:** Changed to PauliX measurement where ⟨X⟩ = sin(input)·sin(θ), giving non-zero gradients.

#### 2. Ring Entanglement → Barren Plateau
**Severity: HIGH**

Full CNOT ring on 8 qubits caused exponential gradient decay:
- θ gradients: ~10⁻¹⁰ (effectively zero)
- Model stuck at random chance (50%)

**Fix:** Disabled entanglement by default. Can be re-enabled for specific use cases.

#### 3. Double Binarization Killed Gradients
**Severity: HIGH**

The forward pass had redundant binarization:
```python
h_bin = binarize_ste(h)  # STE preserves gradients ✓
h01 = (h_bin + 1) / 2
h_bin01 = binarize_features(h01)  # (x > threshold).float() kills gradients ✗
```

**Fix:** Removed redundant `binarize_features()` call after STE.

#### 4. Float64/Float32 Dtype Mismatch
**Severity: MEDIUM**

PennyLane returns float64 tensors, but PyTorch Linear layers use float32:
```
RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float
```

**Fix:** Added `.float()` conversion after stacking expectation values.

### New Features

#### Continuous Encoding Mode
For image data where binarization loses information:
```python
h = torch.tanh(self.fc1(x)) * np.pi  # Maps to [-π, π]
```

Results on MNIST 0-vs-1:
- Binary encoding: 54% (information loss)
- Continuous encoding: 60.3% (matches classical)

#### Gradient Diagnostics
Added gradient checking for barren plateau detection:
```python
print(f'theta_quantum.grad norm: {model.theta_quantum.grad.norm():.6f}')
```

### Benchmark Results

#### Synthetic Parity Task
| Model | Accuracy |
|-------|----------|
| Classical | 46.1% |
| BQNN (a=1.0) | **52.3%** (+6.2%) |

#### MNIST 0-vs-1 (4×4)
| Model | Accuracy |
|-------|----------|
| Classical | 60.3% |
| BQNN (continuous) | 60.3% |
| BQNN (binary) | 54.0% |

**Key Finding:** No quantum advantage on linearly-separable tasks. Quantum layer ≈ tanh nonlinearity when entanglement disabled.

### Files Changed

| File | Changes |
|------|---------|
| `bqnn/model.py` | PauliX measurement, disabled entanglement, dtype fix, removed double binarization |
| `README.md` | Added continuous encoding docs, benchmark results, limitations |
| `CHANGELOG.md` | This update |

---

## [0.2.0] - 2024-12-08

### Critical Bug Fixes (Infrastructure)

#### 1. `pyproject.toml` - Invalid Dependency Syntax
**Severity: CRITICAL**
```diff
- [project.dependencies]
- numpy = "*"
+ dependencies = [
+     "numpy>=1.24.0",
```

#### 2. Missing `bqnn/utils/__init__.py`
**Severity: CRITICAL**

The utils subpackage had no `__init__.py`, causing `ImportError`.

#### 3. `model.py` - Closure Capture Bug
**Severity: CRITICAL**
```python
# BEFORE: noise_pairs captured at definition time
def quantum_layer(angle_vec):
    for _ in range(self.noise_pairs):  # Captured at __init__!

# AFTER: Pass as argument
def quantum_layer(angle_vec, theta, noise_pairs, noise_angle):
    for _ in range(noise_pairs):  # Dynamic at call time
```

#### 4. `data.py` - Identical Train/Test Sets
**Severity: CRITICAL**
```python
# BEFORE: Same seed every call
torch.manual_seed(0)  # Always 0!

# AFTER: Configurable seed
def get_synthetic_dataset(..., seed: Optional[int] = None):
```

#### 5. `data.py` - Test Set From Training Split
**Severity: HIGH**
```python
# BEFORE
ds = MNIST(root="./data", train=True, ...)  # Always training!

# AFTER
ds = MNIST(root="./data", train=train, ...)  # Configurable
```

#### 6. `training.py` - Optimizer Recreation
**Severity: HIGH**
```python
# BEFORE: New optimizer every epoch (kills momentum)
def train_epoch(model, loader, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)

# AFTER: Accept external optimizer
def train_epoch(model, loader, lr, optimizer=None):
```

#### 7. `plots.py` - Heatmap Axis Labels Swapped
**Severity: MEDIUM**

### New Features

- **Trainer class** with persistent optimizer state
- **TrainingConfig** dataclass
- **DeepBQNNModel** with multi-layer circuits
- **Straight-Through Estimator** for classical baseline
- **Comprehensive metrics** (F1, confusion matrix, gradient stats)
- **get_train_test_loaders** convenience function

---

## [0.1.0] - Initial Release

- Basic BQNN implementation
- Synthetic and MNIST data loaders
- Simple training loop 
