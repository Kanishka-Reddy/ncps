import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np


class LTCCell(nn.Module):
    def __init__(
        self,
        wiring,
        in_features=None,
        input_mapping="affine",
        output_mapping="affine",
        ode_unfolds=6,
        epsilon=1e-8,
        implicit_param_constraints=False,
    ):
        """A `Liquid time-constant (LTC) <https://ojs.aaai.org/index.php/AAAI/article/view/16936>`_ cell.

        .. Note::
            This is an RNNCell that process single time-steps. To get a full RNN that can process sequences see `ncps.pytorch.LTC`.


        :param wiring:
        :param in_features:
        :param input_mapping:
        :param output_mapping:
        :param ode_unfolds:
        :param epsilon:
        :param implicit_param_constraints:
        """
        super(LTCCell, self).__init__()
        if in_features is not None:
            wiring.build(in_features)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown number of input features. Please pass the parameter 'in_features' or call the 'wiring.build()'."
            )
        self.make_positive_fn = (
            nn.Softplus() if implicit_param_constraints else nn.Identity()
        )
        self._implicit_param_constraints = implicit_param_constraints
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        self._wiring = wiring
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        self._clip = torch.nn.ReLU()
        self._allocate_parameters()

    @property
    def state_size(self):
        return self._wiring.units

    @property
    def sensory_size(self):
        return self._wiring.input_dim

    @property
    def motor_size(self):
        return self._wiring.output_dim

    @property
    def output_size(self):
        return self.motor_size

    @property
    def synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    @property
    def sensory_synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    def add_weight(self, name, init_value, requires_grad=True):
        param = torch.nn.Parameter(init_value, requires_grad=requires_grad)
        self.register_parameter(name, param)
        return param

    def _get_init_value(self, shape, param_name):
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return torch.ones(shape) * minval
        else:
            return torch.rand(*shape) * (maxval - minval) + minval

    def _allocate_parameters(self):
        self._params = {}
        self._params["gleak"] = self.add_weight(
            name="gleak", init_value=self._get_init_value((self.state_size,), "gleak")
        )
        self._params["vleak"] = self.add_weight(
            name="vleak", init_value=self._get_init_value((self.state_size,), "vleak")
        )
        self._params["cm"] = self.add_weight(
            name="cm", init_value=self._get_init_value((self.state_size,), "cm")
        )
        self._params["sigma"] = self.add_weight(
            name="sigma",
            init_value=self._get_init_value(
                (self.state_size, self.state_size), "sigma"
            ),
        )
        self._params["mu"] = self.add_weight(
            name="mu",
            init_value=self._get_init_value((self.state_size, self.state_size), "mu"),
        )
        self._params["w"] = self.add_weight(
            name="w",
            init_value=self._get_init_value((self.state_size, self.state_size), "w"),
        )
        self._params["erev"] = self.add_weight(
            name="erev",
            init_value=torch.Tensor(self._wiring.erev_initializer()),
        )
        self._params["sensory_sigma"] = self.add_weight(
            name="sensory_sigma",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_sigma"
            ),
        )
        self._params["sensory_mu"] = self.add_weight(
            name="sensory_mu",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_mu"
            ),
        )
        self._params["sensory_w"] = self.add_weight(
            name="sensory_w",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_w"
            ),
        )
        self._params["sensory_erev"] = self.add_weight(
            name="sensory_erev",
            init_value=torch.Tensor(self._wiring.sensory_erev_initializer()),
        )

        self._params["sparsity_mask"] = self.add_weight(
            "sparsity_mask",
            torch.Tensor(np.abs(self._wiring.adjacency_matrix)),
            requires_grad=False,
        )
        self._params["sensory_sparsity_mask"] = self.add_weight(
            "sensory_sparsity_mask",
            torch.Tensor(np.abs(self._wiring.sensory_adjacency_matrix)),
            requires_grad=False,
        )

        if self._input_mapping in ["affine", "linear"]:
            self._params["input_w"] = self.add_weight(
                name="input_w",
                init_value=torch.ones((self.sensory_size,)),
            )
        if self._input_mapping == "affine":
            self._params["input_b"] = self.add_weight(
                name="input_b",
                init_value=torch.zeros((self.sensory_size,)),
            )

        if self._output_mapping in ["affine", "linear"]:
            self._params["output_w"] = self.add_weight(
                name="output_w",
                init_value=torch.ones((self.motor_size,)),
            )
        if self._output_mapping == "affine":
            self._params["output_b"] = self.add_weight(
                name="output_b",
                init_value=torch.zeros((self.motor_size,)),
            )

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = torch.unsqueeze(v_pre, -1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self.make_positive_fn(
            self._params["sensory_w"]
        ) * self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )
        sensory_w_activation = (
            sensory_w_activation * self._params["sensory_sparsity_mask"]
        )

        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]

        # Reduce over dimension 1 (=source sensory neurons)
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        # cm/t is loop invariant
        cm_t = self.make_positive_fn(self._params["cm"]) / (
            elapsed_time / self._ode_unfolds
        )

        # Unfold the multiply ODE multiple times into one RNN step
        w_param = self.make_positive_fn(self._params["w"])
        for t in range(self._ode_unfolds):
            w_activation = w_param * self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )

            w_activation = w_activation * self._params["sparsity_mask"]

            rev_activation = w_activation * self._params["erev"]

            # Reduce over dimension 1 (=source neurons)
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            gleak = self.make_positive_fn(self._params["gleak"])
            numerator = cm_t * v_pre + gleak * self._params["vleak"] + w_numerator
            denominator = cm_t + gleak + w_denominator

            # Avoid dividing by 0
            v_pre = numerator / (denominator + self._epsilon)

        return v_pre

    def _map_inputs(self, inputs):
        if self._input_mapping in ["affine", "linear"]:
            inputs = inputs * self._params["input_w"]
        if self._input_mapping == "affine":
            inputs = inputs + self._params["input_b"]
        return inputs

    def _map_outputs(self, state):
        output = state
        if self.motor_size < self.state_size:
            output = output[:, 0 : self.motor_size]  # slice

        if self._output_mapping in ["affine", "linear"]:
            output = output * self._params["output_w"]
        if self._output_mapping == "affine":
            output = output + self._params["output_b"]
        return output

    def apply_weight_constraints(self):
        if not self._implicit_param_constraints:
            # In implicit mode, the parameter constraints are implemented via
            # a softplus function at runtime
            self._params["w"].data = self._clip(self._params["w"].data)
            self._params["sensory_w"].data = self._clip(self._params["sensory_w"].data)
            self._params["cm"].data = self._clip(self._params["cm"].data)
            self._params["gleak"].data = self._clip(self._params["gleak"].data)

    def forward(self, inputs, states, elapsed_time=1.0):

        # Regularly sampled mode (elapsed time = 1 second)
        inputs = self._map_inputs(inputs)

        next_state = self._ode_solver(inputs, states, elapsed_time)

        outputs = self._map_outputs(next_state)

        return outputs, next_state


class AdaptiveLTCCell(LTCCell):
    def __init__(self, *args, time_penalty: float = 0.01, ponder_epsilon: float = 0.01, time_limit: int = 100, **kwargs):
        super().__init__(*args, **kwargs)

        self.time_penalty = time_penalty
        self.ponder_epsilon = ponder_epsilon
        self.time_limit = time_limit

        # Halting unit: a simple linear layer to decide whether to halt or not
        self.halting_unit = nn.Sequential(
            nn.Linear(self.state_size, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs, state, elapsed_time=1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Map inputs
        inputs = self._map_inputs(inputs)

        # Initialize ACT related variables
        remaining_budget = 1.0 - self.ponder_epsilon
        ponder_cost = 0
        step_count = 0

        while step_count < self.time_limit and remaining_budget > 0:
            # Run one step of the ODE solver
            next_state = self._ode_solver(inputs, state, elapsed_time)

            # Check halting condition
            p_continue = self.halting_unit(next_state).squeeze()
            step_halt = (remaining_budget - p_continue) > 0
            remaining_budget -= p_continue.item()

            # Update ponder cost and step count
            ponder_cost += self.time_penalty * step_halt
            step_count += 1

            # Update state if not halted
            state = next_state if step_halt else state

        # Map outputs
        outputs = self._map_outputs(state)

        return outputs, state, ponder_cost, step_count


import torch
import torch.nn as nn
import ncps.wirings  # Assuming you have the 'ncps' package for wirings

class AdaptiveLTC(nn.Module):
    def __init__(
        self,
        input_size: int,
        units,
        return_sequences: bool = True,
        batch_first: bool = True,
        mixed_memory: bool = False,
        input_mapping="affine",
        output_mapping="affine",
        ode_unfolds=6,
        epsilon=1e-8,
        implicit_param_constraints=True,
        time_penalty: float = 0.01,
        ponder_epsilon: float = 0.01,
        time_limit: int = 100,
    ):
        super(AdaptiveLTC, self).__init__()

        self.input_size = input_size
        self.wiring_or_units = units
        self.batch_first = batch_first
        self.return_sequences = return_sequences

        # Determine the wiring to be used
        if isinstance(units, ncps.wirings.Wiring):
            wiring = units
        else:
            wiring = ncps.wirings.FullyConnected(units)

        # Initialize the AdaptiveLTCCell with the provided parameters
        self.rnn_cell = AdaptiveLTCCell(
            wiring=wiring,
            in_features=input_size,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            ode_unfolds=ode_unfolds,
            epsilon=epsilon,
            implicit_param_constraints=implicit_param_constraints,
            time_penalty=time_penalty,
            ponder_epsilon=ponder_epsilon,
            time_limit=time_limit,
        )
        self._wiring = wiring
        self.use_mixed = mixed_memory
        if self.use_mixed:
            self.lstm = nn.LSTMCell(input_size, self.state_size)

    @property
    def state_size(self):
        return self._wiring.units

    def forward(self, input, hx=None):
        device = input.device
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0

        if not is_batched:
            input = input.unsqueeze(batch_dim)

        batch_size, seq_len = input.size(batch_dim), input.size(seq_dim)

        # Initialize hidden state if not provided
        if hx is None:
            hx = torch.zeros((batch_size, self.state_size), device=device)

        output_sequence = []
        ponder_cost_sequence = []
        step_count_sequence = []

        for t in range(seq_len):
            if self.batch_first:
                inputs = input[:, t]
            else:
                inputs = input[t]

            h_out, hx, ponder_cost, step_count = self.rnn_cell(inputs, hx)

            ponder_cost_tensor = torch.tensor([ponder_cost], device=device).unsqueeze(
                0)  # Unsqueeze to add a new dimension
            step_count_tensor = torch.tensor([step_count], device=device).unsqueeze(
                0)  # Unsqueeze to add a new dimension

            if self.return_sequences:
                output_sequence.append(h_out)
                ponder_cost_sequence.append(ponder_cost_tensor)
                step_count_sequence.append(step_count_tensor)

        if self.return_sequences:
            outputs = torch.cat(output_sequence, dim=seq_dim)  # Concatenate along the sequence dimension
            ponder_costs = torch.cat(ponder_cost_sequence, dim=seq_dim)  # Concatenate along the sequence dimension
            step_counts = torch.cat(step_count_sequence, dim=seq_dim)  # Concatenate along the sequence dimension
        else:
            outputs = h_out
            ponder_costs = ponder_cost_sequence[-1].squeeze(0)  # Get the last element and remove the extra dimension
            step_counts = step_count_sequence[-1].squeeze(0)  # Get the last element and remove the extra dimension

        return outputs, hx, ponder_costs, step_counts

