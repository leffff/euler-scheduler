from typing import Tuple, Any, Optional

import torch
from torch import Tensor


def get_time_coefficients(timesteps: torch.Tensor, ndim: int) -> torch.Tensor:
    return timesteps.reshape((timesteps.shape[0], *([1] * (ndim - 1))))


class FlowMatchingEulerSchedulerOutput:
    """
    Output class for the scheduler's `step` function output.

    Args:
        model_output (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Output of the model `f(x_t, t, c)`
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Current sample `x_t`
        timesteps  (`torch.Tensor` of shape `(batch_size)`)
            Current deoising step `t`
        h (`float`)
            1 / num_inference_steps coefficient, used to scale model prediction in Euler Method
    """

    def __init__(self, model_output, sample, timesteps, h):
        self.model_output = model_output
        self.sample = sample
        self.timesteps = timesteps
        self.h = h

    @property
    def prev_sample(self) -> torch.Tensor:
        return self.sample + self.h * self.model_output

    @property
    def pred_original_sample(self) -> torch.Tensor:
        return self.sample + (1 - get_time_coefficients(self.timesteps, self.model_output.ndim)) * self.model_output


class FlowMatchingEulerScheduler:
    """
    `FlowMatchingEulerScheduler` is a scheduler for training and inferencing Conditional Flow Matching models (CFMs)

    Args:
        num_inference_steps (`int`, defaults to 100):
            The number of steps on inference.
    """

    def __init__(self, num_inference_steps: int = 100):
        self.timesteps = None
        self.num_inference_steps = None
        self.h = None

        if num_inference_steps is not None:
            self.set_timesteps(num_inference_steps)

    @staticmethod
    def add_noise(original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Add noise to the given sample

        Args:
            original_samples (`torch.Tensor`):
                The original sample that is to be noised
            noise (`torch.Tensor`):
                The noise that is used to noise the image
            timesteps (`torch.Tensor`):
                Timesteps used to create linear interpolation `x_t = t * x_1 + (1 - t) * x_0`
        """

        t = get_time_coefficients(timesteps, original_samples.ndim)

        noised_sample = t * original_samples + (1 - t) * noise

        return noised_sample

    def set_timesteps(self, num_inference_steps: int = 100) -> None:
        """
        Set number of inference steps (Euler intagration steps)

        Args:
            num_inference_steps (`int`, defaults to 100):
                The number of steps on inference.
        """

        self.num_inference_steps = num_inference_steps
        self.h = 1 / num_inference_steps
        self.timesteps = torch.arange(0, 1, self.h)

    def step(self, model_output: torch.Tensor, timesteps: torch.Tensor, sample: torch.Tensor,
             return_dict: bool = True) -> FlowMatchingEulerSchedulerOutput | tuple[Tensor, Tensor]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        step = FlowMatchingEulerSchedulerOutput(model_output=model_output, sample=sample, timesteps=timesteps, h=self.h)

        if return_dict:
            return step

        return sample, step.prev_sample

    @staticmethod
    def get_velocity(original_samples: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            original_samples (`torch.Tensor`):
                The original sample that is to be noised
            noise (`torch.Tensor`):
                The noise that is used to noise the image

        Returns:
            `torch.Tensor`
        """

        return original_samples - noise

    @staticmethod
    def scale_model_input(sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
         Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
         current timestep.

        Args:
            sample (`torch.Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """

        return sample
