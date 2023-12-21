import torch
import torch.nn as nn
import numpy as np
import sys
from typing import List, Optional, Tuple,Union
import ncps

class AdaptiveCellWrapper(nn.Module):
    """
    Wraps an RNN cell to add adaptive computation time.

    Note that the cell will need an input size of 1 plus the desired input size, to
    allow for the extra first-step flag input.

    Parameters
    ----------
    cell
        The cell to wrap.
    time_penalty
        How heavily to penalize the model for thinking too long. Tau in Graves 2016.
    initial_halting_bias
        Value to initialize the halting unit's bias to. Recommended to set this
        to a negative number to prevent long ponder sequences early in training.
    ponder_epsilon
        When the halting values sum to more than 1 - ponder_epsilon, stop computation.
        Used to enable halting on the first step.
    time_limit
        Hard limit for how many substeps any computation can take. Intended to prevent
        overly-long computation early-on. M in Graves 2016.
    """

    time_penalty: float
    ponder_epsilon: float
    time_limit: int

    _cell: torch.nn.modules.RNNCellBase
    _halting_unit: torch.nn.Module

    def __init__(
        self,
        cell: torch.nn.RNNCellBase,
        time_penalty: float,
        initial_halting_bias: float = -1.0,
        ponder_epsilon: float = 1e-2,
        time_limit: int = 100,
    ):
        super().__init__()

        if time_penalty <= 0:
            raise ValueError("time_penalty must be positive.")
        if ponder_epsilon < 0 or ponder_epsilon >= 1:
            raise ValueError(
                "ponder_epsilon must be between 0 (inclusive) and 1 (exclusive)"
            )

        self.time_penalty = time_penalty
        self.ponder_epsilon = ponder_epsilon
        self.time_limit = time_limit

        self._cell = cell
        self._halting_unit = torch.nn.Sequential(
            torch.nn.Linear(cell.state_size, 1),
            torch.nn.Flatten(),  # type: ignore
            torch.nn.Sigmoid(),
        )

        torch.nn.init.constant_(self._halting_unit[0].bias, initial_halting_bias)

    def forward(
        self, inputs: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute one timestep of the RNN, which may correspond to several internal steps.

        Parameters
        ----------
        inputs
            Tensor containing input features.

            *Shape: (batch, input_size)*
        hidden
            Initial hidden value for the wrapped cell. If not provided, relies on
            the wrapped cell to provide its own initial value.

            *Shape: (batch, hidden_size)*

        Returns
        -------
        next_hiddens : torch.Tensor
            The hidden state for this timestep.

            *Shape: (batch, hidden_size)*
        ponder_cost : torch.Tensor
            The ponder cost for this timestep.

            *Shape: () (scalar)*
        ponder_steps : torch.Tensor
            The number of ponder steps each element in the batch took.

            *Shape: (batch)*
        """
        batch_size = inputs.size(0)
        budget = torch.ones((batch_size, 1), device=inputs.device) - self.ponder_epsilon

        # Accumulate intermediate values throughout the ponder sequence
        total_hidden = torch.zeros(
            [batch_size, self._cell.state_size], device=inputs.device
        )
        total_remainder = torch.zeros_like(budget)
        total_steps = torch.zeros_like(budget)

        # Extend input with first-step flag
        first_inputs = torch.cat([inputs, torch.ones_like(budget)], dim=1)
        rest_inputs = torch.cat([inputs, torch.zeros_like(budget)], dim=1)

        # Sum of halting values for all steps except the last
        halt_accum = torch.zeros_like(budget)
        continuing_mask = torch.ones_like(budget, dtype=torch.bool)

        for step in range(self.time_limit - 1):
            step_inputs = first_inputs if step == 0 else rest_inputs
            hidden = self._cell(step_inputs, hidden)

            step_halt = self._halting_unit(hidden)
            masked_halt = continuing_mask * step_halt

            with torch.no_grad():
                halt_accum += masked_halt

            # Select indices ending at this step
            ending_mask = continuing_mask.bitwise_and(halt_accum + step_halt > budget)
            continuing_mask = continuing_mask.bitwise_and(ending_mask.bitwise_not())
            total_steps += continuing_mask

            # 3 cases, computed in parallel by masking batch elements:
            # - Continuing computation: weight new values by the halting probability
            # - Ending at this step: weight new values by the remaining budget
            # - Ended previously: no new values (accumulate zero)
            masked_remainder = ending_mask * (1 - halt_accum)
            combined_mask = masked_halt + masked_remainder

            total_hidden = total_hidden + (combined_mask * hidden)
            total_remainder = total_remainder + masked_halt

            # If all batch indices are done, stop iterations early
            if not continuing_mask.any().item():
                break

        else:  # Some elements ran past the hard limit
            # continuing_mask now selects for these elements
            masked_remainder = continuing_mask * (1 - halt_accum)
            total_hidden = total_hidden + (masked_remainder * hidden)

        # Equal gradient to the true cost; maximize all halting values except the last
        ponder_cost = -1 * self.time_penalty * total_remainder.mean()

        return total_hidden, ponder_cost, total_steps + 1

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

    # @property
    # def hidden_size(self):
    #     return self._wiring.units

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

class AdaptiveLTCCell(AdaptiveCellWrapper):
    """
    An adaptive-time variant of LTCCell.

    Parameters
    ----------
    ltc_cell
        An instance of LTCCell.
    time_penalty
        How heavily to penalize the model for thinking too long. Tau in Graves 2016.
    initial_halting_bias
        Value to initialize the halting unit's bias to. Recommended to set this
        to a negative number to prevent long ponder sequences early in training.
    ponder_epsilon
        When the halting values sum to more than 1 - ponder_epsilon, stop computation.
        Used to enable halting on the first step.
    time_limit
        Hard limit for how many substeps any computation can take. Intended to prevent
        overly-long computation early-on. M in Graves 2016.
    """

    def __init__(
        self,
        ltc_cell: LTCCell,
        time_penalty: float,
        initial_halting_bias: float = -1.0,
        ponder_epsilon: float = 1e-2,
        time_limit: int = 100,
    ):
        super().__init__(
            ltc_cell, time_penalty, initial_halting_bias, ponder_epsilon, time_limit
        )


