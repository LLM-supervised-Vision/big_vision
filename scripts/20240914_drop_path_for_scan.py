import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List

class EncoderBlock(nn.Module):
    hidden_dim: int
    drop_path_rate: float

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        y = nn.Dense(self.hidden_dim)(x)
        y = nn.relu(y)
        if not deterministic:
            y = drop_path(y, self.drop_path_rate)
        return x + y

def drop_path(x, drop_prob: float = 0.0):
    """Simplified drop path function for demonstration."""
    if drop_prob == 0.0:
        return x
    keep_prob = 1 - drop_prob
    mask = jax.random.bernoulli(jax.random.PRNGKey(0), p=keep_prob, shape=(x.shape[0], 1, 1))
    return x * mask / keep_prob

class Encoder(nn.Module):
    num_layers: int
    hidden_dim: int
    max_drop_path_rate: float

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # Convert drop_path_rates to a JAX array
        drop_path_rates = jnp.linspace(0, self.max_drop_path_rate, self.num_layers) if self.max_drop_path_rate > 0 else jnp.zeros(self.num_layers)

        # Define the scanned function
        def scan_fn(carry, drop_path_rate):
            x = carry
            block = EncoderBlock(self.hidden_dim, drop_path_rate)
            x = block(x, deterministic)
            return x, x

        # Use nn.scan to apply the encoder blocks
        _, outputs = nn.scan(
            EncoderBlock,
            variable_broadcast="params",
            split_rngs={"params": False, "dropout": True},
            in_axes=0,
            length=self.num_layers,
        )(self.hidden_dim, 
        name="encoderblock"
        )(x, drop_path_rates, deterministic)

        return outputs[-1]  # Return the final layer's output

# Test the Encoder
def test_encoder():
    batch_size, seq_len, input_dim, hidden_dim = 2, 10, 32, 32
    num_layers = 4
    max_drop_path_rate = 0.2

    x = jnp.ones((batch_size, seq_len, input_dim))
    
    encoder = Encoder(num_layers, hidden_dim, max_drop_path_rate)
    params = encoder.init(jax.random.PRNGKey(0), x)
    
    # Test with deterministic=False to see drop path in action
    output = encoder.apply(params, x, deterministic=False)
    print("Encoder output shape:", output.shape)

    # You can also print some values to verify that the output is changing
    print("Sample output values:", output[0, 0, :5])

if __name__ == "__main__":
    test_encoder()