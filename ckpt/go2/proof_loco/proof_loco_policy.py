from collections.abc import Sequence
from functools import partial
from pathlib import Path

import numpy as np
import onnxruntime as ort

from unitree_deploy.obs import ObservationContext
from unitree_deploy.policy.base_policy import BasePolicy, ExecutionProvider, ObservationRegistry
from unitree_deploy.utils.import_utils import import_symbol


class ProofLocoPolicy(BasePolicy):
    """WebDemo depth encoder and recurrent actor."""

    def __init__(
        self,
        policy_yaml_path: str | Path,
        *,
        providers: Sequence[ExecutionProvider] | None = None,
        observation_types: ObservationRegistry | None = None,
    ) -> None:
        super().__init__(policy_yaml_path, providers=providers, observation_types=observation_types)
        encoder_providers = self.session.get_provider_options()
        self.encoder = ort.InferenceSession(
            str(self.policy_yaml_path.parent / self.config["depth_encoder_path"]),
            providers=[(name, encoder_providers[name]) for name in self.session.get_providers()],
        )
        actor_inputs = {item.name: item for item in self.session.get_inputs()}
        self.recurrent_inputs = {}
        for name in ("h_in", "gate_in", "gate_valid_in", "gate_open_in"):
            info = actor_inputs[name]
            dtype = np.bool_ if info.type == "tensor(bool)" else np.float32
            self.recurrent_inputs[name] = np.zeros(info.shape, dtype=dtype)
        self.recurrent_outputs = ["h_out", "gate_out", "gate_valid_out", "gate_open_out"]
        self.target_q_clip = self.config["target_q_clip"]
        depth_spec = self.config["depth_observation"]
        params = dict(depth_spec["params"])
        source_class = import_symbol(params.pop("source_class"), self.policy_yaml_path.parent)
        params["open_source"] = partial(source_class, **params.pop("source_params"))
        self.depth = self._build_observation({**depth_spec, "params": params})
        depth_shape = (1, self.depth.history_len, *self.depth.image_shape)
        encoder_shape = tuple(self.encoder.get_inputs()[0].shape)
        if depth_shape != encoder_shape:
            self.depth.close()
            raise ValueError(f"depth observation has shape {depth_shape}, encoder expects {encoder_shape}")
        self.depth_shape = depth_shape

    def reset(self) -> None:
        super().reset()
        self.depth.reset()
        for state in self.recurrent_inputs.values():
            state.fill(0)

    def close(self) -> None:
        self.depth.close()

    def compute_target_q(self, context: ObservationContext) -> np.ndarray:
        self.depth.update(context)
        target = super().compute_target_q(context)
        return np.clip(target, *self.target_q_clip, out=target)

    def _infer_action(self, obs_vector: np.ndarray) -> np.ndarray:
        depth = self.depth.ordered_buffer().reshape(self.depth_shape)
        (latent,) = self.encoder.run(["depth_image_latent"], {"depth_image": depth})
        actor_obs = np.concatenate((obs_vector[None, :], latent), axis=1)
        action, *states = self.session.run(
            [self.action_output_name, *self.recurrent_outputs],
            {self.input_name: actor_obs, **self.recurrent_inputs},
        )
        for name, state in zip(self.recurrent_inputs, states):
            self.recurrent_inputs[name] = state
        return action.reshape(-1)
