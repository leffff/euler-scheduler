import torch


def get_time_coefficients(timesteps: torch.Tensor, ndim: int) -> torch.Tensor:
    return timesteps.reshape((timesteps.shape[0], *([1] * (ndim - 1))))


class FlowMatchingEulerSchedulerOutput:
    def __init__(self, model_output, sample, timesteps, h):
        self.model_output = model_output
        self.sample = sample
        self.timesteps = timesteps
        self.h = h

    @property
    def prev(self):
        return self.sample + self.h * self.model_output

    @property
    def original_sample(self):
        return self.sample + (1 - get_time_coefficients(self.timesteps, self.model_output.ndim)) * self.model_output


class FlowMatchingEulerScheduler:
    def __init__(self, num_inference_steps: int = None):
        self.timesteps = None
        self.num_inference_steps = None
        self.h = None

        if num_inference_steps is not None:
            self.set_timesteps(num_inference_steps)

    @staticmethod
    def add_noise(original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t = get_time_coefficients(timesteps, original_samples.ndim)

        noised_sample = t * original_samples + (1 - t) * noise

        return noised_sample

    def set_timesteps(self, num_inference_steps: int = 100) -> None:
        self.num_inference_steps = num_inference_steps
        self.h = 1 / num_inference_steps
        self.timesteps = torch.arange(0, 1, self.h)

    def step(self, model_output: torch.Tensor, sample: torch.Tensor,
             timesteps: torch.Tensor) -> FlowMatchingEulerSchedulerOutput:
        return FlowMatchingEulerSchedulerOutput(model_output=model_output, sample=sample, timesteps=timesteps, h=self.h)

    @staticmethod
    def get_velocity(original_samples: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return original_samples - noise
