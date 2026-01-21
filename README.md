# Arnold Transformer

A PyTorch implementation of a GPT-style Transformer built from scratch. This project serves as a testbed for the experimental **Arnold Activation** function.

## 🌟 Arnold Activation

The Arnold Activation is based on the **Circle Map** (also known as the Arnold Circle Map), a classic example from chaos theory that exhibits phase-locking and chaotic behavior.

### Mathematical Formulation

The activation function applies the circle map iteration to the input $x$:

$$ f(x) = (x + \Omega - \frac{K}{2\pi} \sin(2\pi x)) \pmod 1 $$

Where:
- $x$: The input value (pre-activation).
- $\Omega$ (**Omega**): The natural rotation number or driving frequency. It determines the inherent shift of the phase. In our implementation, defaults to the Golden Ratio ($\phi \approx 0.618$).
- $K$: The **coupling strength** or non-linearity parameter. This is a **learnable parameter** in our network.

### Dynamics & Chaos

The behavior of the map changes based on $K$:
- **$K = 0$**: The map is a simple linear rotation: $x_{n+1} = x_n + \Omega \pmod 1$.
- **$0 < K < 1$**: The map becomes non-linear but remains invertible (diffeomorphism). Mode-locking (Arnold Tongues) can occur.
- **$K = 1$**: The critical line. The map becomes non-invertible at specific points.
- **$K > 1$**: The map becomes non-invertible and can exhibit **chaotic dynamics**.

### Lyapunov Exponent

To measure the "chaos" introduced by the layer, we calculate the **Lyapunov Exponent** ($\lambda$). For a 1D map $f(x)$, this is the average logarithm of the absolute derivative:

$$ \lambda \approx \frac{1}{N} \sum \ln |f'(x)| $$
$$ f'(x) = 1 - K \cos(2\pi x) $$

- $\lambda < 0$: Stable, predictable behavior (attractors).
- $\lambda > 0$: **Chaotic behavior** (sensitive dependence on initial conditions).

We use this value in the **Lyapunov Governor** to curb the learning rate if the network enters a deeply chaotic regime ($\lambda$ becomes too positive), preventing gradients from exploding or vanishing unpredictably.

### Lyapunov Governor

Training a network with chaotic components can be unstable. To mitigate this, we implement a **Lyapunov Governor**:
- It monitors the maximum Lyapunov exponent across all Arnold layers.
- If the system becomes too chaotic (high $\lambda$), the governor **dynamically reduces the learning rate**.
- **Formula**: $LR_{new} = LR_{base} \times e^{-\max(0, \lambda) \cdot \beta}$

## 🚀 Usage

### Installation

This project uses `uv` for dependency management, but standard pip works too.

```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install torch numpy tiktoken
```

### Data Preparation

Place your text training data (`.txt` files) in the `dataset/` directory. The data loader will automatically find and concatenate them.

### Training

Run the training script:

```bash
uv run train.py
```

### Configuration

You can configure the model architecture and training hyperparameters in `train.py`.

**Layer-wise Activation configuration:**
You can mix `ReLU` and `Arnold` layers. For example, to put Arnold activations in the middle layers of a 16-layer model:

```python
# in train.py
n_layer = 16
activation_types = ['relu'] * n_layer
middle = n_layer // 2
activation_types[middle] = 'arnold'
# ...
```

## 📂 Project Structure

- `model.py`: The GPT architecture and `ArnoldActivation` implementation.
- `train.py`: Training loop, hyperparameter config, and Lyapunov governor logic.
- `data_loader.py`: Handles loading text files and tokenization (using `tiktoken`).
