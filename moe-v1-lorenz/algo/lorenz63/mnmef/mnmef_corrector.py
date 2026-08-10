"""MNMEF CorrTerms analysis step adapted from the DALearning implementation."""

import torch
import torch.nn as nn

from mnmef_localization import create_loc_mat, pairwise_distances
from mnmef_networks import SetTransformer, Simple_MLP
from mnmef_utils import mean0


class MNMEFCorrTerms(nn.Module):
    """Learned EnKF analysis with oracle per-step observation covariance."""

    def __init__(self, state_dim, clamp):
        super().__init__()
        self.state_dim = state_dim
        self.clamp = clamp
        representation_dim = 64
        self.input_dim = 3 * state_dim + 2 * representation_dim
        self.local_input_dim = 2 * representation_dim

        indices = torch.arange(state_dim)
        state_observation_distance = pairwise_distances(
            indices[:, None], indices[:, None], domain=(state_dim,)
        )
        observation_distance = pairwise_distances(
            indices[:, None], indices[:, None], domain=(state_dim,)
        )
        distance_values = torch.unique(
            torch.cat([state_observation_distance.flatten(), observation_distance.flatten()])
        )
        self.register_buffer("Lvy", state_observation_distance)
        self.register_buffer("Lyy", observation_distance)
        self.register_buffer("diff_dist", distance_values)

        self.model = Simple_MLP(self.input_dim, 2 * state_dim, num_hidden_layers=2)
        self.st_model1 = SetTransformer(
            input_dim=2 * state_dim,
            num_heads=8,
            num_inds=16,
            output_dim=2 * representation_dim,
            hidden_dim=64,
            num_layers=2,
            freeze_WQ=True,
        )
        self.infl_model = Simple_MLP(
            state_dim + 2 * representation_dim, state_dim, num_hidden_layers=2
        )
        self.local_model = Simple_MLP(
            self.local_input_dim, len(distance_values), num_hidden_layers=2
        )

    def forward(self, forecast, observation, observation_covariance, observation_std):
        predicted_observation = torch.atan(forecast)
        batch, ensemble_size, state_dim = forecast.shape
        perturbed_noise = mean0(observation_std * torch.randn_like(predicted_observation))
        innovation = observation - predicted_observation - perturbed_noise

        observation_mean = predicted_observation.mean(1, keepdim=True).expand_as(predicted_observation)
        forecast_mean = forecast.mean(1, keepdim=True).expand_as(forecast)
        representation = self.st_model1(torch.cat([forecast, predicted_observation], -1))
        repeated_representation = representation[:, None].expand(-1, ensemble_size, -1)

        network_input = torch.cat(
            [forecast, predicted_observation, observation.expand(-1, ensemble_size, -1),
             repeated_representation], -1
        ).reshape(-1, self.input_dim)
        inflation_input = torch.cat([forecast, repeated_representation], -1).reshape(
            -1, state_dim + repeated_representation.shape[-1]
        )
        correction = self.model(network_input).view(batch, ensemble_size, -1)
        inflated_forecast = forecast + self.infl_model(inflation_input).view_as(forecast)
        corrected_state_anomalies = forecast - forecast_mean + correction[:, :, :state_dim]
        corrected_observation_anomalies = (
            predicted_observation - observation_mean + correction[:, :, state_dim:]
        )

        localization = torch.sigmoid(self.local_model(representation)) * 2.0
        state_observation_localization = create_loc_mat(localization, self.diff_dist, self.Lvy)
        observation_localization = create_loc_mat(localization, self.diff_dist, self.Lyy)
        gain_left = torch.bmm(
            corrected_state_anomalies.transpose(1, 2), corrected_observation_anomalies
        ) * state_observation_localization
        gain_right = torch.bmm(
            corrected_observation_anomalies.transpose(1, 2), corrected_observation_anomalies
        ) * observation_localization
        gain_right = gain_right + observation_covariance * (ensemble_size - 1)
        gain = torch.bmm(gain_left, torch.inverse(gain_right))
        analysis = inflated_forecast + torch.bmm(innovation, gain.transpose(1, 2))
        return analysis.clamp(-self.clamp, self.clamp)
