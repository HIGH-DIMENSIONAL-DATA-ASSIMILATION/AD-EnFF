import math

import torch
import torch.nn as nn

from TimeSeriesModel import detach_state, make_carrier
from common.enff_utils import get_marginal_vf


STATE_SCALE = 50.0
KEY_WIDTH = 16

# The five lawyers, in the fixed order used for signals, GRUs, and outputs throughout this file.
# (Indices 0-2 are ensemble-aware; 3-4 are mean-only. See how each signal is built in forward().)
LAWYER_ANALYSIS_CORRECTION = 0    # previous posterior - previous prior      (obs-free)
LAWYER_PRIOR_INNOVATION = 1       # y - h(current prior)                     (uses current y)
LAWYER_POSTERIOR_INNOVATION = 2   # y - h(previous posterior)                (uses current y)
LAWYER_OBSERVATION_JUMP = 3       # y - previous observation                 (uses current y)
LAWYER_PROCESS_JUMP = 4           # current prior - previous posterior       (obs-free)


def guided_velocity(zt, z0, z1, time, sigma_min, observation, guidance):
    """Analytical EnFF flow velocity at ODE time `time`, for the elementwise arctan sensor.

    The flow transports each particle `zt` from the start `z0` (previous posterior) toward
    the end `z1` (current, corrected prior). Two parts are added:

    1. `marginal`: the unconditioned flow-matching velocity (where the particle would go with
       no observation). This is the plain analytical EnFF transport.
    2. observation guidance: a pull that nudges the particle so its predicted observation
       matches the real observation `y`.

    `guidance` (= learned lambda, one value per state dim) sets how hard we pull toward `y`.
    """
    # Unconditioned transport velocity from the flow-matching field.
    marginal = get_marginal_vf(zt, z0, z1, time, sigma_min)
    # Straight-line estimate of where this particle lands at t=1 (its predicted x1).
    endpoint = zt + (1.0 - time) * marginal
    # Gradient of the arctan observation energy 0.5*(atan(x)-y)^2 w.r.t. the predicted endpoint:
    #   d/dx [0.5*(atan(x)-y)^2] = (atan(x) - y) * 1/(1+x^2).
    # observation is [B, D]; unsqueeze to [B, 1, D] to broadcast over the particle axis.
    residual_gradient = (torch.atan(endpoint) - observation.unsqueeze(1)) / (1.0 + endpoint.square())
    # Descend that energy (minus sign), scaled per-dimension by the learned guidance strength.
    # Clamp guards against blow-up if a denominator ever gets tiny.
    return marginal - guidance * residual_gradient.clamp(-1000.0, 1000.0)


class AdaptiveENFF(nn.Module):
    """AD-EnFF used in the reported Lorenz experiments.

    The model learns a per-state guidance strength and a per-state prior mean
    correction. The first three lawyers receive the means and variances of
    their particlewise fields.
    """

    def __init__(self, dim_x, attention=False, nt_ode=12, state_clamp=60.0):
        super().__init__()
        if attention:
            raise ValueError("AdaptiveENFF is plain-only; attention is not supported")
        self.dim_x = dim_x
        self.nt_ode = nt_ode
        self.state_clamp = state_clamp
        self.n_lawyers = 5

        width = 128            # hidden width of every GRU/Mamba/MLP block
        layers = 4             # depth of each recurrent block
        # Lawyers 0-2 receive particle-field means and variances. Lawyers 3-4
        # receive one and two mean-only fields, respectively.
        lawyer_inputs = [4 * dim_x, 2 * dim_x, 2 * dim_x, dim_x, 2 * dim_x]

        self.lambda_lawyers = nn.ModuleList(
            make_carrier("gru", size, width, width, width, 32, n_layers=layers)
            for size in lawyer_inputs
        )
        self.correction_lawyers = nn.ModuleList(
            make_carrier("gru", size, width, width, width, 32, n_layers=layers)
            for size in lawyer_inputs
        )
        self.lambda_key_heads = nn.ModuleList(
            nn.Linear(width, KEY_WIDTH) for _ in range(self.n_lawyers)
        )
        self.lambda_value_heads = nn.ModuleList(
            nn.Linear(width, width) for _ in range(self.n_lawyers)
        )
        self.lambda_key_norm = nn.LayerNorm(KEY_WIDTH)
        self.lambda_judge = make_carrier(
            "gru", self.n_lawyers * KEY_WIDTH, width, width, width, 32, n_layers=layers
        )
        self.lambda_retainer = make_carrier(
            "mamba", width * (self.n_lawyers + 1), width,
            mamba_d_model=width, d_state=16, headdim=32, n_layers=layers,
        )

        self.correction_key_heads = nn.ModuleList(
            nn.Linear(width, KEY_WIDTH) for _ in range(self.n_lawyers)
        )
        self.correction_value_heads = nn.ModuleList(
            nn.Linear(width, width) for _ in range(self.n_lawyers)
        )
        self.correction_key_norm = nn.LayerNorm(KEY_WIDTH)
        self.correction_judge = make_carrier(
            "gru", self.n_lawyers * KEY_WIDTH, width, width, width, 32, n_layers=layers
        )
        self.correction_retainer = make_carrier(
            "mamba", width * (self.n_lawyers + 1), width,
            mamba_d_model=width, d_state=16, headdim=32, n_layers=layers,
        )

        # Guidance head -> one lambda per state dim. Last layer starts near zero, and lambda_bias
        # is chosen so softplus(0 + lambda_bias) = 0.05, i.e. training begins at the analytical
        # EnFF's fixed guidance strength and moves away from there.
        self.lambda_head = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, dim_x),
        )
        nn.init.normal_(self.lambda_head[-1].weight, 0.0, 0.01)
        nn.init.zeros_(self.lambda_head[-1].bias)
        self.lambda_bias = math.log(math.expm1(0.05))   # inverse-softplus of 0.05

        # Correction head -> one delta per state dim. Zero-initialised so training starts from
        # delta = 0 (no prior shift) and only learns a correction if it helps.
        self.correction_head = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, dim_x)
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)
        self.reset_cache()

    def reset_cache(self):
        # Wipe all recurrent memory and cached history. Call at the START of each trajectory so
        # one trajectory's state never leaks into the next.
        self._lambda_lawyer_states = [None] * self.n_lawyers
        self._correction_lawyer_states = [None] * self.n_lawyers
        self._lambda_judge_state = self._lambda_retainer_state = None
        self._correction_judge_state = self._correction_retainer_state = None
        self._previous_observation = None
        self._previous_prior = self._previous_posterior = None
        self._previous_correction_signals = None

    def detach_memory(self):
        # Detach (but keep) all memory at the TBPTT window boundary: values carry forward, but
        # gradients do not flow past this point, bounding backprop to one window.
        self._lambda_lawyer_states = [
            detach_state(state) for state in self._lambda_lawyer_states
        ]
        self._correction_lawyer_states = [
            detach_state(state) for state in self._correction_lawyer_states
        ]
        self._lambda_judge_state = detach_state(self._lambda_judge_state)
        self._lambda_retainer_state = detach_state(self._lambda_retainer_state)
        self._correction_judge_state = detach_state(self._correction_judge_state)
        self._correction_retainer_state = detach_state(self._correction_retainer_state)
        if self._previous_correction_signals is not None:
            self._previous_correction_signals = [
                signal.detach() for signal in self._previous_correction_signals
            ]

    @staticmethod
    def _ensemble_mean_var(fields):
        """Summarise per-particle fields [B, J, D] by their ensemble mean and variance.

        Returns each field's mean and variance, concatenated along the feature axis.
        """
        means = [field.mean(1) for field in fields]
        variances = [field.var(1, unbiased=False) for field in fields]
        return torch.cat(means + variances, -1)

    def forward(self, z1, z0, y):
        # z1: current forecast prior ensemble  [B, J particles, D dims]
        # z0: previous posterior ensemble      [B, J, D]  (flow start)
        # y : current observation              [B, D]     (already in arctan/observation space)
        batch, _, dimension = z1.shape
        prior_mean = z1.mean(1)                          # ensemble-mean of the current prior, [B, D]

        # Pull the previous cycle's cached ensembles / observation; on the first step of a
        # trajectory (cache is None) fall back to zeros so nothing below special-cases step 0.
        previous_observation = self._previous_observation if self._previous_observation is not None else torch.zeros_like(y)
        previous_prior = self._previous_prior if self._previous_prior is not None else torch.zeros_like(z1)
        previous_posterior = self._previous_posterior if self._previous_posterior is not None else torch.zeros_like(z1)
        previous_posterior_mean = previous_posterior.mean(1)   # derived on demand, not cached separately

        # ---- Build one signal per lawyer. h(.) is the arctan sensor; an "innovation" is the
        # observation minus a predicted observation. ----
        #
        # Lawyers 0-2 receive per-particle fields reduced to their means and variances.
        particle_fields = [
            # 0) analysis correction (posterior - prior), in state space and observation space
            [(previous_posterior - previous_prior) / STATE_SCALE,
             torch.atan(previous_posterior) - torch.atan(previous_prior)],
            # 1) current prior innovation: y - h(each prior particle)
            [y.unsqueeze(1) - torch.atan(z1)],
            # 2) previous posterior innovation: y - h(each previous posterior particle)
            [y.unsqueeze(1) - torch.atan(previous_posterior)],
        ]
        particle_signals = [self._ensemble_mean_var(fields) for fields in particle_fields]

        # Lawyers 3-4 are mean-only: one mean-based vector each, no particle version.
        process_move = (prior_mean - previous_posterior_mean) / STATE_SCALE
        observation_jump = y - previous_observation                              # 3) how the observation moved
        process_jump = torch.cat(                                                # 4) how the forecast moved the state
            [process_move, torch.atan(prior_mean) - torch.atan(previous_posterior_mean)], -1
        )

        signals = particle_signals + [observation_jump, process_jump]

        # ---- Guidance (lambda) branch: read the current evidence and decide the pull strength. ----
        # Each lawyer is a GRU that carries its own memory across cycles. It emits a hidden
        # `output`, from which we take a small "key" (its argument) and a wider "value" (its evidence).
        lambda_keys = []
        lambda_values = []
        lambda_inputs = zip(self.lambda_lawyers, signals)
        for index, (lawyer, current_signal) in enumerate(lambda_inputs):
            output, self._lambda_lawyer_states[index] = lawyer.step(
                current_signal, self._lambda_lawyer_states[index]
            )
            lambda_keys.append(self.lambda_key_norm(self.lambda_key_heads[index](output)))
            lambda_values.append(self.lambda_value_heads[index](output))

        # Judge (GRU) weighs the five keys into a verdict; retainer (Mamba) holds long memory
        # and combines the verdict with the values into the final "advice" vector.
        lambda_verdict, self._lambda_judge_state = self.lambda_judge.step(
            torch.cat(lambda_keys, -1), self._lambda_judge_state
        )
        lambda_advice, self._lambda_retainer_state = self.lambda_retainer.step(
            torch.cat([lambda_verdict] + lambda_values, -1), self._lambda_retainer_state
        )

        # ---- Correction (delta) branch: shift the whole prior to remove forecast-model bias. ----
        # Delay observation-dependent inputs before they enter the correction lawyers. Their
        # recurrent states therefore never receive evidence containing the current observation.
        previous_correction_signals = self._previous_correction_signals
        if previous_correction_signals is None:
            previous_correction_signals = [torch.zeros_like(signal) for signal in signals]

        correction_keys = []
        correction_values = []
        correction_inputs = zip(self.correction_lawyers, signals)
        for index, (lawyer, current_signal) in enumerate(correction_inputs):
            if index in (LAWYER_ANALYSIS_CORRECTION, LAWYER_PROCESS_JUMP):
                correction_input = current_signal
            else:
                correction_input = previous_correction_signals[index]
            output, self._correction_lawyer_states[index] = lawyer.step(
                correction_input, self._correction_lawyer_states[index]
            )
            correction_keys.append(
                self.correction_key_norm(self.correction_key_heads[index](output))
            )
            correction_values.append(
                self.correction_value_heads[index](output)
            )
        # Same judge/retainer structure as the lambda branch, but its own weights.
        correction_verdict, self._correction_judge_state = self.correction_judge.step(
            torch.cat(correction_keys, -1), self._correction_judge_state
        )
        correction_advice, self._correction_retainer_state = self.correction_retainer.step(
            torch.cat([correction_verdict] + correction_values, -1),
            self._correction_retainer_state,
        )
        # delta: one mean-shift per state dim, shared across particles (zero-initialised head,
        # so training starts from delta = 0). Add it to the whole prior before the ODE.
        prior_correction = self.correction_head(correction_advice)
        corrected_prior = z1 + prior_correction.unsqueeze(1)
        self._previous_correction_signals = list(signals)

        # lambda: one positive guidance strength per state dimension. softplus keeps it > 0, and
        # lambda_bias is set so that at init (head output ~ 0) lambda ~ 0.05, i.e. we start from
        # the analytical EnFF's fixed guidance value. Observation noise is never given to the model.
        guidance = torch.nn.functional.softplus(
            self.lambda_head(lambda_advice) + self.lambda_bias
        ).view(batch, 1, dimension)

        # ---- Run the guided EnFF ODE: transport z0 -> corrected_prior over t: 0 -> 1. ----
        state = z0 + 0.1 * torch.randn_like(z0)                                   # start from previous posterior + small jitter
        time_mesh = torch.linspace(0.0, 1.0, self.nt_ode + 1, device=z1.device)   # nt_ode Euler steps
        for index in range(self.nt_ode):
            time = time_mesh[index].item()
            step = time_mesh[index + 1] - time_mesh[index]                        # dt
            sigma_min = 0.1
            velocity = guided_velocity(state, z0, corrected_prior, time, sigma_min, y, guidance)
            state = state + step * velocity.clamp(-1000.0, 1000.0)                # explicit Euler update

        # Cache this cycle's quantities (detached) for next cycle's signals. Detaching here stops
        # gradients flowing through the physical state history; the recurrent memories carry it instead.
        self._previous_observation = y.detach()
        self._previous_prior = z1.detach()
        self._previous_posterior = state.detach()
        return {
            "posterior": state.clamp(-self.state_clamp, self.state_clamp),        # final posterior ensemble [B, J, D]
            "lam": guidance.squeeze(1),                                           # learned lambda [B, D] (for logging)
            "prior_delta": prior_correction,                                      # learned delta [B, D] (for logging/penalty)
        }
