"""
BQNN Model implementation.

Provides the core quantum-classical hybrid neural network with tunable
quantumness parameter and configurable noise injection.
"""

from typing import Optional, List, Callable

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

from .quantization import binarize_features, features_to_angles
from .classical_reference import binarize_ste


class BQNNModel(nn.Module):
    """
    Binarized Quantum Neural Network (BQNN) model.
    
    Architecture:
        1. Classical linear layer + sign activation
        2. Quantum layer:
           - RX encoding of binary features
           - Trainable RZ phase layer
           - Ring entanglement (CNOT chain)
           - Optional U U† noise pairs
           - Pauli-Z expectation measurements
        3. Linear classifier on measurement outcomes
    
    The quantumness parameter `a` controls the angle mapping:
        - a=0: Fixed angle mapping (classical-like behavior)
        - a>0: Increased angular spread (more quantum variance)
    
    Args:
        n_features: Input feature dimension
        n_hidden: Hidden layer width (also determines n_qubits if not specified)
        n_classes: Number of output classes
        n_qubits: Number of qubits (defaults to n_hidden)
        a: Quantumness parameter
        dev_name: PennyLane device name
        shots: Number of measurement shots (None for exact expectation)
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        n_classes: int,
        n_qubits: Optional[int] = None,
        a: float = 0.5,
        dev_name: str = "default.qubit",
        shots: Optional[int] = None,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.a = a
        self.shots = shots

        if n_qubits is None:
            n_qubits = n_hidden
        self.n_qubits = n_qubits

        # Classical layers
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc_out = nn.Linear(n_qubits, n_classes)

        # Trainable quantum phase parameters
        self.theta_quantum = nn.Parameter(
            0.1 * torch.randn(n_qubits),
            requires_grad=True,
        )

        # Noise configuration - stored as buffers for proper state management
        # Using register_buffer for non-trainable state that should persist
        self.register_buffer("_noise_pairs", torch.tensor(0, dtype=torch.int32))
        self.register_buffer("_noise_angle", torch.tensor(0.1, dtype=torch.float32))
        
        # Ansatz configuration
        self.entanglement_type = "ring"  # 'ring', 'full', 'linear'
        self.n_layers = 1

        # Create device
        self.dev = qml.device(dev_name, wires=self.n_qubits, shots=shots)
        
        # Build quantum layer - we pass noise params explicitly to avoid closure issues
        self._build_quantum_layer()

    def _build_quantum_layer(self):
        """
        Build the quantum layer qnode.
        
        NOTE: We pass noise parameters as arguments to the qnode to avoid
        the closure capture problem where self.noise_pairs/noise_angle
        would be captured at definition time rather than runtime.
        """
        n_qubits = self.n_qubits
        
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def quantum_layer(
            angle_vec: torch.Tensor,
            theta_quantum: torch.Tensor,
            noise_pairs: int,
            noise_angle: float,
        ):
            """
            BQNN-style quantum circuit.
            
            Args:
                angle_vec: Input encoding angles [n_qubits]
                theta_quantum: Trainable phase parameters [n_qubits]
                noise_pairs: Number of U U† noise injection pairs
                noise_angle: Rotation angle for noise gates
                
            Returns:
                List of Pauli-Z expectation values
            """
            # 1. Input encoding layer
            for i in range(n_qubits):
                qml.RX(angle_vec[i], wires=i)

            # 2. Trainable phase layer
            for i in range(n_qubits):
                qml.RZ(theta_quantum[i], wires=i)

            # 3. Entangling layer (ring topology) - OPTIONAL
            # Disabled by default to avoid barren plateaus
            # if n_qubits > 1:
            #     for i in range(n_qubits - 1):
            #         qml.CNOT(wires=[i, i + 1])
            #     qml.CNOT(wires=[n_qubits - 1, 0])

            # 4. Noise injection: U U† pairs
            # These should ideally cancel but accumulate errors on real hardware
            for _ in range(noise_pairs):
                for j in range(n_qubits):
                    qml.RX(noise_angle, wires=j)
                    qml.RX(-noise_angle, wires=j)

            # 5. Measurement layer - use PauliX to get θ dependence
            # (PauliZ after RZ has zero gradient since RZ only adds phase)
            return [qml.expval(qml.PauliX(w)) for w in range(n_qubits)]

        self._quantum_layer_fn = quantum_layer

    @property
    def noise_pairs(self) -> int:
        return int(self._noise_pairs.item())
    
    @property
    def noise_angle(self) -> float:
        return float(self._noise_angle.item())

    def set_quantumness(self, a: float):
        """Set the quantumness parameter."""
        self.a = float(a)

    def set_noise(self, n_pairs: int, angle: float = 0.1):
        """
        Configure noise injection.
        
        Args:
            n_pairs: Number of U U† gate pairs to inject
            angle: Rotation angle for noise gates
        """
        self._noise_pairs.fill_(int(n_pairs))
        self._noise_angle.fill_(float(angle))

    def quantum_forward(self, angle_vec: torch.Tensor) -> torch.Tensor:
        """
        Execute quantum layer on a batch of angle vectors.
        
        Args:
            angle_vec: Input angles [batch_size, n_qubits]
            
        Returns:
            Expectation values [batch_size, n_qubits]
        """
        # Get current noise parameters (now properly dynamic)
        noise_pairs = self.noise_pairs
        noise_angle = self.noise_angle
        
        # Process batch - vectorized where possible
        expvals = []
        for i in range(angle_vec.shape[0]):
            ev = self._quantum_layer_fn(
                angle_vec[i],
                self.theta_quantum,
                noise_pairs,
                noise_angle,
            )
            # Stack list of expvals into tensor
            if isinstance(ev, list):
                ev = torch.stack([e if isinstance(e, torch.Tensor) else torch.tensor(e, dtype=torch.float32) 
                                 for e in ev])
            expvals.append(ev)
        
        return torch.stack(expvals, dim=0).to(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the BQNN.
        
        Pipeline:
            1. Linear transform + sign activation (binarization)
            2. Map to {0,1} → rotation angles
            3. Quantum layer → expectation values  
            4. Linear classifier → logits
            
        Args:
            x: Input features [batch_size, n_features]
            
        Returns:
            Class logits [batch_size, n_classes]
        """
        # Classical preprocessing with STE for gradient flow
        h = self.fc1(x)
        h_bin = binarize_ste(h)  # STE allows gradients through sign()
        
        # Map from {-1, +1} to {0, 1}
        h01 = ((h_bin + 1.0) / 2.0).clamp(0.0, 1.0)

        # h01 is already binary after STE, skip redundant binarize_features
        # Select first n_qubits features
        h_bin01 = h01[:, :self.n_qubits]
        
        # Convert to rotation angles
        angle_vec = features_to_angles(h_bin01, a=self.a)

        # Quantum layer
        expvals = self.quantum_forward(angle_vec)

        # Classical classifier
        logits = self.fc_out(expvals.to(torch.float32))
        return logits

    def get_circuit_info(self) -> dict:
        """Return circuit statistics for analysis."""
        return {
            "n_qubits": self.n_qubits,
            "n_trainable_params": self.theta_quantum.numel(),
            "noise_pairs": self.noise_pairs,
            "noise_angle": self.noise_angle,
            "quantumness": self.a,
            "entanglement": self.entanglement_type,
            "shots": self.shots,
        }


class DeepBQNNModel(BQNNModel):
    """
    Extended BQNN with multiple quantum layers.
    
    Allows deeper quantum circuits with repeated encoding-entanglement blocks.
    """
    
    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        n_classes: int,
        n_qubits: Optional[int] = None,
        n_layers: int = 2,
        a: float = 0.5,
        dev_name: str = "default.qubit",
        shots: Optional[int] = None,
    ):
        super().__init__(
            n_features=n_features,
            n_hidden=n_hidden,
            n_classes=n_classes,
            n_qubits=n_qubits,
            a=a,
            dev_name=dev_name,
            shots=shots,
        )
        self.n_layers = n_layers
        
        # Additional trainable parameters for each layer
        self.theta_layers = nn.ParameterList([
            nn.Parameter(0.1 * torch.randn(self.n_qubits))
            for _ in range(n_layers - 1)
        ])
        
        self._build_deep_quantum_layer()
    
    def _build_deep_quantum_layer(self):
        """Build multi-layer quantum circuit."""
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def deep_quantum_layer(
            angle_vec: torch.Tensor,
            theta_params: List[torch.Tensor],
            noise_pairs: int,
            noise_angle: float,
        ):
            # Initial encoding
            for i in range(n_qubits):
                qml.RX(angle_vec[i], wires=i)
            
            # Repeated layers
            for layer_idx, theta in enumerate(theta_params):
                # Phase layer
                for i in range(n_qubits):
                    qml.RZ(theta[i], wires=i)
                
                # Entanglement
                if n_qubits > 1:
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    qml.CNOT(wires=[n_qubits - 1, 0])
            
            # Noise injection
            for _ in range(noise_pairs):
                for j in range(n_qubits):
                    qml.RX(noise_angle, wires=j)
                    qml.RX(-noise_angle, wires=j)
            
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]
        
        self._deep_quantum_layer_fn = deep_quantum_layer
    
    def quantum_forward(self, angle_vec: torch.Tensor) -> torch.Tensor:
        """Execute deep quantum layer."""
        noise_pairs = self.noise_pairs
        noise_angle = self.noise_angle
        
        # Collect all theta parameters
        all_thetas = [self.theta_quantum] + list(self.theta_layers)
        
        expvals = []
        for i in range(angle_vec.shape[0]):
            ev = self._deep_quantum_layer_fn(
                angle_vec[i],
                all_thetas,
                noise_pairs,
                noise_angle,
            )
            if isinstance(ev, list):
                ev = torch.stack([e if isinstance(e, torch.Tensor) else torch.tensor(e, dtype=torch.float32)
                                 for e in ev])
            expvals.append(ev)
        
        return torch.stack(expvals, dim=0).to(torch.float32)
