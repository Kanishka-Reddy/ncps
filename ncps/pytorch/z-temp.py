class _CellUnrollWrapper(torch.nn.Module):
    """
    Wraps an adaptive cell into a torch.nn.RNN-style interface.

    NOTE: because the adaptive cell requires dynamic computation, this does not
    support packed sequences and will not be any more efficient than running the
    cell in a loop (e.g. you don't get the full cuDNN benefits).
    This mainly exists for convenience.

    Parameters
    ----------
    layer_cells
        One cell per layer in the network.
    batch_first
        If True, expects the first dimension of each sequence to be the batch axis
        and the second to be the sequence axis.
    dropout
        Amount of dropout to apply to the output of each layer except the last.
    """

    batch_first: bool
    _layer_cells: torch.nn.ModuleList
    _dropout_layer: Optional[torch.nn.Dropout]

    def __init__(
        self,
        layer_cells: List[AdaptiveCellWrapper],
        batch_first: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        if not layer_cells:
            raise ValueError("layer_cells must be nonempty.")

        self.batch_first = batch_first
        self._layer_cells = torch.nn.ModuleList(layer_cells)

        if dropout:
            self._dropout_layer = torch.nn.Dropout(dropout)
        else:
            self._dropout_layer = None

    def _apply_layers(
        self, inputs: torch.Tensor, layer_hiddens: List[Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Apply the stack of RNN cells (one per layer) to one timestep of inputs
        and hiddens.

        Parameters
        ----------
        inputs
            A tensor of inputs for this timestep.

            *Shape: (batch, dims)*.
        layer_hiddens
            One (optional) hidden state per layer.

        Returns
        -------
        outputs : torch.Tensor
            The output of the layer stack for this timestep.
        next_hiddens : List[torch.Tensor]
            The layer hidden states for the next timestep.
        ponder_cost : torch.Tensor
            The total ponder cost for this timestep.
        """
        total_ponder_cost = torch.tensor(0.0, device=inputs.device)
        next_hiddens = []

        for i, (cell, hidden) in enumerate(zip(self._layer_cells, layer_hiddens)):
            if self._dropout_layer and i > 0:  # Applies dropout between every 2 layers
                inputs = self._dropout_layer(inputs)

            inputs, ponder_cost, _ = cell(inputs, hidden)
            next_hiddens.append(inputs)
            total_ponder_cost = total_ponder_cost + ponder_cost

        return inputs, next_hiddens, total_ponder_cost  # type: ignore

    def forward(
        self, inputs: torch.Tensor, hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        inputs
            Input to the network.

            *Shape: (seq_len, batch, input_size)*, or *(batch, seq_len, input_size)*
            if batch_first is True.
        hidden
            Initial hidden state for each element in the batch, as a tensor of

            *Shape: (num_layers \* num_directions, batch, hidden_size)*

        Returns
        -------
        output : torch.Tensor
            The output features from the last layer of the RNN for each timestep.

            *Shape: (seq_len, batch, hidden_size)*, or *(batch, seq_len, hidden_size)*
            if batch_first is True.
        hidden : torch.Tensor
            The hidden state for the final step.

            *Shape: (num_layers, batch, hidden_size)*
        ponder_cost : torch.Tensor
            The total ponder cost for this sequence.

            *Shape: ()*
        """
        if self.batch_first:
            inputs = torch.transpose(inputs, 0, 1)

        timesteps, _, _ = inputs.shape

        if hidden is None:
            layer_hiddens = [None for _ in self._layer_cells]
        else:
            layer_hiddens = [h.squeeze(0) for h in hidden.split(1)]

        total_ponder_cost = torch.tensor(0.0, device=inputs.device)
        outputs = []

        for timestep in range(timesteps):
            output, layer_hiddens, ponder_cost = self._apply_layers(  # type: ignore
                inputs[timestep, :, :], layer_hiddens  # type: ignore
            )
            outputs.append(output)
            total_ponder_cost = total_ponder_cost + ponder_cost

        all_outputs = torch.stack(outputs)  # Stacks timesteps
        all_hiddens = torch.stack(layer_hiddens)  # type: ignore  # Stacks layers

        if self.batch_first:
            all_outputs = torch.transpose(all_outputs, 0, 1)

        return all_outputs, all_hiddens, total_ponder_cost



class AdaptiveLTC(_CellUnrollWrapper):
    """
    An adaptive-time variant of the LTC RNN.

    Similar parameters as the original LTC, with additional parameters for adaptive computation.
    """

    def __init__(
            self,
            input_size: int,
            units,
            time_penalty: float,
            initial_halting_bias: float = -1.0,
            ponder_epsilon: float = 1e-2,
            time_limit: int = 100,
            batch_first: bool = False,
            dropout: float = 0.0,
            # Additional parameters from LTC
            return_sequences: bool = True,
            mixed_memory: bool = False,
            input_mapping="affine",
            output_mapping="affine",
            ode_unfolds=6,
            epsilon=1e-8,
            implicit_param_constraints=True,
    ):
        if isinstance(units, ncps.wirings.Wiring):
            wiring = units
        else:
            wiring = ncps.wirings.FullyConnected(units)

        # Create a single layer AdaptiveCellWrapper with LTCCell
        ltc_cell = LTCCell(
            wiring=wiring,
            in_features=input_size,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            ode_unfolds=ode_unfolds,
            epsilon=epsilon,
            implicit_param_constraints=implicit_param_constraints,
        )

        adaptive_ltc_cell = AdaptiveCellWrapper(
            ltc_cell,
            time_penalty,
            initial_halting_bias,
            ponder_epsilon,
            time_limit
        )

        # If using mixed memory, you would need to handle that separately
        if mixed_memory:
            raise NotImplementedError("Mixed memory is not implemented for AdaptiveLTC")

        # Initialize the _CellUnrollWrapper with the adaptive cell
        super().__init__([adaptive_ltc_cell], batch_first, dropout)
        self.return_sequences = return_sequences

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

    def forward(self, input, hx=None, timespans=None):
        # Forward method adapted from LTC
        output, hx, ponder_cost = super().forward(input, hx)

        if not self.return_sequences:
            output = output[-1]

        return output, hx, ponder_cost