import torch
from ncps.pytorch.AdaptiveNewEquationLTC5 import LTCCell, AdaptiveLTCCell
from ncps.wirings import AutoNCP

# Define the wiring
wiring = AutoNCP(64, 8)  # 28 neurons, 4 outputs

# Create an instance of AdaptiveLTCCell
adaptive_ltc_cell = AdaptiveLTCCell(
    wiring=wiring,
    in_features=32,  # Assuming the input feature size is 20
    time_penalty=0.05,
    ponder_epsilon=1e-2,
    time_limit=10
)

# Create a dummy input tensor
dummy_input = torch.randn(1, 32)  # Input size of 20

# Initialize hidden state
hidden_state = torch.zeros(1, 64)  # Batch size of 1, hidden state size of 28

# Perform a forward pass
output, hidden_state, ponder_cost, ponder_steps = adaptive_ltc_cell(dummy_input, hidden_state)

print("Forward pass successful.")
print("Output shape:", output.shape)
print("Ponder cost:", ponder_cost.item())
print("Ponder steps:\n\n", ponder_steps)

"""
Now for the Adaptive LTC itself
"""

import torch
from ncps.pytorch.AdaptiveLTC4 import AdaptiveLTC
from ncps.wirings import AutoNCP

# Define model parameters
input_size = 20  # Input feature size
units = 28  # Number of neurons
output_size = 4  # Number of outputs
sequence_length = 10  # Length of the input sequence
batch_size = 1  # Batch size

# Create the wiring
wiring = AutoNCP(units, output_size)

# Initialize the AdaptiveLTC model
adaptive_ltc = AdaptiveLTC(
    input_size=input_size,
    units=wiring,
    return_sequences=True,
    batch_first=True
)

# Create a dummy input tensor (batch size, sequence length, input size)
dummy_input = torch.randn(batch_size, sequence_length, input_size)

# Forward pass through the model
outputs, hx, ponder_costs, step_counts = adaptive_ltc(dummy_input)

# Print results
print("Output shape:", outputs.shape)
print("Hidden state shape:", hx.shape)
print("Ponder costs shape:", ponder_costs)
print("Step counts shape:", step_counts)

# note to self, compare shape of outputs with normal LTC to see if they are the same


