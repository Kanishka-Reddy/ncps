# from ncps.pytorch import LTC
# import torch
# from ncps.pytorch.atc_ltc import ACTLTC
# from ncps.pytorch import AdaptiveLTC


# rnn = AdaptiveLTC(20,50, 0.01)
# x = torch.randn(2, 3, 20) # (batch, time, features)
# h0 = torch.zeros(2,50) # (batch, units)
# output, hn = rnn(x,h0)
# print(output)

# rnn2 = ACTLTC(20, 50)
# x = torch.randn(2, 3, 20) # (batch, time, features)
# h0 = torch.zeros(2,50) # (batch, units)
# output, hn, total_ponder_cost = rnn2(x,h0)
# print(total_ponder_cost)

import torch
# from ncps.pytorch.AdaptiveLTC2 import AdaptiveCellWrapper, LTCCell, AdaptiveLTCCell  # Import or define LTCCell here
# from ncps.wirings import AutoNCP
#
# wiring = AutoNCP(28, 4) # 28 neurons, 4 outputs

# Create an instance of LTCCell with appropriate parameters
# Note: Replace 'your_wiring' and other parameters with actual values suitable for your use case
# import torch
# from ncps.pytorch.AdaptiveLTC3 import LTCCell, AdaptiveLTCCell
# from ncps.wirings import AutoNCP
#
# # Define the wiring
# wiring = AutoNCP(28, 4)  # 28 neurons, 4 outputs
#
# # Create an instance of LTCCell with appropriate parameters
# # ltc_cell = LTCCell(wiring=wiring, in_features=20)
#
# # Create an instance of AdaptiveLTCCell
# adaptive_ltc_cell = AdaptiveLTCCell(
#     wiring=wiring,
#     in_features = 20,
#     time_penalty=0.01,  # Example value, adjust as necessary
#     initial_halting_bias=-1.0,
#     ponder_epsilon=1e-2,
#     time_limit=100
# )
#
# # Create a dummy input tensor
# # Adjust the shape according to your expectations (batch size, input size + 1 for the first-step flag)
# # Create a dummy input tensor
# dummy_input = torch.randn(1, 20)  # Original input size of 20
#   # Example: batch size of 1, input size of 20 + 1 for the flag
#
# # Assuming the hidden state is initialized to zeros
# # Adjust the hidden state size according to your LTCCell configuration
# hidden_state = torch.zeros(1, 28)  # Example: batch size of 1, hidden state size of 28 (number of neurons)
#
# # Perform a forward pass
# output, ponder_cost, ponder_steps = adaptive_ltc_cell(dummy_input, hidden_state)
#
# print("Forward pass successful.")
# print("Output shape:", output.shape)
# print("Ponder cost:", ponder_cost.item())
# print("Ponder steps:", ponder_steps.item())


"""
aLTC3
"""
import torch
from ncps.pytorch.AdaptiveLTC4 import LTCCell, AdaptiveLTCCell, AdaptiveLTC
from ncps.wirings import AutoNCP
#
# # Define the wiring
# wiring = AutoNCP(28, 4)  # 28 neurons, 4 outputs
#
# # Create an instance of LTCCell with appropriate parameters
# ltc_cell = LTCCell(wiring=wiring, in_features=20)
#
# # Create an instance of AdaptiveLTCCell
# adaptive_ltc_cell = AdaptiveLTCCell(
#     ltc_cell=ltc_cell,  # Pass the LTCCell instance here
#     time_penalty=0.01,
#     initial_halting_bias=-1.0,
#     ponder_epsilon=1e-2,
#     time_limit=100
# )
#
# # Create a dummy input tensor
# dummy_input = torch.randn(1, 21)  # Input size of 20 + 1 for the first-step flag
#
# # Initialize hidden state
# hidden_state = torch.zeros(1, 28)  # Batch size of 1, hidden state size of 28
#
# # Perform a forward pass
# output, ponder_cost, ponder_steps = adaptive_ltc_cell(dummy_input, hidden_state)
#
# print("Forward pass successful.")
# print("Output shape:", output.shape)
# print("Ponder cost:", ponder_cost.item())
# print("Ponder steps:", ponder_steps.item())


"""
ALTC4
"""
import torch
from ncps.wirings import FullyConnected

# Assuming you have already defined AdaptiveLTCCell and LTCCell

# Define a simple wiring
wiring = FullyConnected(10)  # Example wiring with 10 units

# Initialize the AdaptiveLTCCell
cell = AdaptiveLTCCell(wiring=wiring, in_features=5, time_penalty=0.01, ponder_epsilon=0.01, time_limit=100)

# Create dummy input and initial state
batch_size = 1
input_size = 5
state_size = cell.state_size
dummy_input = torch.randn(batch_size, input_size)
initial_state = torch.randn(batch_size, state_size)

# Run the cell
output, new_state, ponder_cost, step_count = cell(dummy_input, initial_state)

print("Output:", output)
print("New State:", new_state)
print("Ponder Cost:", ponder_cost)
print("Step Count:", step_count)

print("NOW FOR THE LTC ITZELF\n")

import torch
from ncps.wirings import FullyConnected  # Assuming 'ncps' package is available

# Define the input size and the wiring for the network
input_size = 5
units = 10  # Number of units in the wiring

# Create a FullyConnected wiring as an example
wiring = FullyConnected(units)

# Initialize the AdaptiveLTC network
adaptive_ltc_net = AdaptiveLTC(
    input_size=input_size,
    units=wiring,
    return_sequences=True,  # Set to False if you only want the final output
    batch_first=True,       # Assuming batch is the first dimension in the input
    time_penalty=0.01,
    ponder_epsilon=0.01,
    time_limit=100
)

# Create dummy input data
batch_size = 1
seq_length = 3  # Length of the sequence
dummy_input = torch.randn(batch_size, seq_length, input_size)

# Run the input through the network
output, final_state, ponder_costs, step_counts = adaptive_ltc_net(dummy_input)

# Print the results
print("Output:", output)
print("Final State:", final_state)
print("Ponder Costs:", ponder_costs)
print("Step Counts:", step_counts)






