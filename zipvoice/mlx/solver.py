from typing import Optional, Union

import mlx.core as mx


def get_time_steps(
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_step: int = 10,
    t_shift: float = 1.0,
) -> mx.array:
    timesteps = mx.linspace(t_start, t_end, num_step + 1, dtype=mx.float32)
    timesteps = t_shift * timesteps / (1 + (t_shift - 1) * timesteps)
    return timesteps


class DiffusionModel:
    def __init__(self, model, func_name: str = "forward_fm_decoder") -> None:
        self.model = model
        self.model_func = getattr(self.model, func_name)

    def __call__(
        self,
        t: mx.array,
        x: mx.array,
        text_condition: mx.array,
        speech_condition: mx.array,
        padding_mask: Optional[mx.array] = None,
        guidance_scale: Union[float, mx.array] = 0.0,
    ) -> mx.array:
        if not isinstance(guidance_scale, mx.array):
            guidance_scale = mx.array(guidance_scale, dtype=t.dtype)

        if float(mx.max(mx.abs(guidance_scale)).item()) == 0.0:
            return self.model_func(
                t=t,
                xt=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
            )

        if t.ndim != 0:
            raise ValueError("Guidance scale inference expects scalar t")

        x = mx.concatenate([x, x], axis=0)
        if padding_mask is not None:
            padding_mask = mx.concatenate([padding_mask, padding_mask], axis=0)

        text_condition = mx.concatenate([mx.zeros_like(text_condition), text_condition], axis=0)

        if float(t.item()) > 0.5:
            speech_condition = mx.concatenate(
                [mx.zeros_like(speech_condition), speech_condition], axis=0
            )
        else:
            guidance_scale = guidance_scale * 2
            speech_condition = mx.concatenate([speech_condition, speech_condition], axis=0)

        data = self.model_func(
            t=t,
            xt=x,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
        )
        half = data.shape[0] // 2
        data_uncond = data[:half]
        data_cond = data[half:]
        return (1 + guidance_scale) * data_cond - guidance_scale * data_uncond


class DistillDiffusionModel(DiffusionModel):
    def __call__(
        self,
        t: mx.array,
        x: mx.array,
        text_condition: mx.array,
        speech_condition: mx.array,
        padding_mask: Optional[mx.array] = None,
        guidance_scale: Union[float, mx.array] = 0.0,
    ) -> mx.array:
        if not isinstance(guidance_scale, mx.array):
            guidance_scale = mx.array(guidance_scale, dtype=t.dtype)
        return self.model_func(
            t=t,
            xt=x,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            guidance_scale=guidance_scale,
        )


class EulerSolver:
    def __init__(self, model, func_name: str = "forward_fm_decoder") -> None:
        self.model = DiffusionModel(model, func_name=func_name)

    def sample(
        self,
        x: mx.array,
        text_condition: mx.array,
        speech_condition: mx.array,
        padding_mask: mx.array,
        num_step: int = 10,
        guidance_scale: Union[float, mx.array] = 0.0,
        t_start: float = 0.0,
        t_end: float = 1.0,
        t_shift: float = 1.0,
    ) -> mx.array:
        timesteps = get_time_steps(
            t_start=t_start,
            t_end=t_end,
            num_step=num_step,
            t_shift=t_shift,
        )
        for step in range(num_step):
            t_cur = timesteps[step]
            t_next = timesteps[step + 1]

            v = self.model(
                t=t_cur,
                x=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                guidance_scale=guidance_scale,
            )

            x_1_pred = x + (1.0 - t_cur) * v
            x_0_pred = x - t_cur * v

            if step < num_step - 1:
                x = (1.0 - t_next) * x_0_pred + t_next * x_1_pred
            else:
                x = x_1_pred
        return x


class DistillEulerSolver(EulerSolver):
    def __init__(self, model, func_name: str = "forward_fm_decoder") -> None:
        self.model = DistillDiffusionModel(model, func_name=func_name)
