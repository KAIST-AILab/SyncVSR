# Copyright 2023 Jungwoo Park
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


class Transform(ABC):
    def __init__(self, *, p: float | None = None):
        self.p = p

    @abstractmethod
    def apply(self, **inputs: Any) -> dict[str, Any]:
        pass

    def __call__(self, **inputs: Any) -> dict[str, Any]:
        if self.p is None or np.random.random() < self.p:
            return self.apply(**inputs)
        return inputs


class Sequential(Transform):
    def __init__(self, *transforms: Callable, **kwargs: Any):
        super().__init__(**kwargs)
        self.transforms = transforms

    def apply(self, **inputs: Any) -> dict[str, Any]:
        for transform in self.transforms:
            inputs = transform(**inputs)
        return inputs


class Identity(Transform):
    def apply(self, **inputs: Any) -> dict[str, Any]:
        return inputs


class GroupApply(Transform):
    def __init__(
        self,
        transforms: Transform | list[Transform],
        lengths: list[int],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if isinstance(transforms, Transform):
            transforms = [transforms] * len(lengths)
        self.transforms = transforms
        self.lengths = lengths

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        outputs = []
        for transform, length in zip(self.transforms, self.lengths):
            outputs.append(transform(landmarks=landmarks[:, :length], **inputs))
            landmarks = landmarks[:, length:]
        landmarks = torch.cat([x["landmarks"] for x in outputs], dim=1)
        return dict(outputs[0], landmarks=landmarks)


class Normalize(Transform):
    def __init__(self, max_value: float | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.max_value = max_value

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        scale = self.max_value or landmarks.nan_to_num(0).std()
        return dict(inputs, landmarks=(landmarks - landmarks.nanmean((0, 1))) / scale)


class LeftCrop(Transform):
    def __init__(self, length: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.length = length

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        return dict(inputs, landmarks=landmarks[: self.length])


class CenterCrop(Transform):
    def __init__(self, length: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.length = length

    def center_crop(self, x: Sequence, total: int) -> Sequence:
        length = int(len(x) * self.length / total)
        start = max((len(x) - length) // 2, 0)
        return x[start : start + length]

    def apply(self, **inputs: Any) -> dict[str, Any]:
        total = inputs["landmarks"].size(0)
        for name in inputs:
            if name == "landmarks" or "label" in name:
                inputs[name] = self.center_crop(inputs[name], total)
        return inputs


class RandomCrop(Transform):
    def __init__(self, length: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.length = length

    def random_crop(self, x: Sequence, start: int, total: int) -> Sequence:
        length = int(len(x) * self.length / total)
        start = int(len(x) * start / total)
        return x[start : start + length]

    def apply(self, **inputs: Any) -> dict[str, Any]:
        total = inputs["landmarks"].size(0)
        start = np.random.randint(max(total - self.length, 1))

        for name in inputs:
            if name == "landmarks" or "label" in name:
                inputs[name] = self.random_crop(inputs[name], start, total)
        return inputs


class Pad(Transform):
    def __init__(self, length: int, value: float = -100.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.length = length
        self.value = value

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        if (padding := self.length - landmarks.size(0)) > 0:
            landmarks = F.pad(landmarks, (0, 0, 0, 0, 0, padding), value=self.value)
        return dict(inputs, landmarks=landmarks)


class HorizontalFlip(Transform):
    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        landmarks = landmarks * torch.tensor([-1.0, 1.0, 1.0])
        return dict(inputs, landmarks=landmarks)


class TimeFlip(Transform):
    def apply(self, **inputs: Any) -> dict[str, Any]:
        inputs["landmarks"] = inputs["landmarks"].flip(0)
        for name in inputs:
            if "label" in name:
                inputs[name] = inputs[name][::-1]
        return inputs


class RandomResample(Transform):
    def __init__(self, limit: float | tuple[float, float] = 0.1, **kwargs: Any):
        super().__init__(**kwargs)
        if not isinstance(limit, Iterable):
            limit = (1 - limit, 1 + limit)
        self.limit = limit

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        # First of all, we need to fill all `nan` values from their previous frame
        # because common linear interpolation uses the previous and next frames to
        # literally interpolate the intermediate values. It makes other frames be
        # invalid and hence we should fill them with proper values.
        x_ff = landmarks.clone()
        for i in range(1, x_ff.size(0)):
            x_ff[i].copy_(x_ff[i].where(~landmarks[i].isnan(), x_ff[i - 1]))

        x_ff = x_ff.flatten(1).transpose(1, 0).unsqueeze(0)
        mask = (~landmarks.isnan()).flatten(1).transpose(1, 0).unsqueeze(0).float()

        # After that, we interpolate the frames as well as their valid mask to infer
        # which positions should be filled with `nan` for the interpolated video.
        scale = np.random.uniform(self.limit[0], self.limit[1])
        x_ff = F.interpolate(x_ff, scale_factor=scale, mode="linear")[0].T
        mask = F.interpolate(mask, scale_factor=scale, mode="linear")[0].T

        x_ff.masked_fill_(mask < 0.5, torch.nan)
        return dict(inputs, landmarks=x_ff.unflatten(1, (-1, 3)))


class CoordinateJitter(Transform):
    def __init__(self, stdev: float = 0.01, **kwargs: Any):
        super().__init__(**kwargs)
        self.stdev = stdev

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        landmarks = landmarks + torch.empty_like(landmarks).normal_(0, self.stdev)
        return dict(inputs, landmarks=landmarks)


class RandomShift(Transform):
    def __init__(self, stdev: float = 0.1, **kwargs: Any):
        super().__init__(**kwargs)
        self.stdev = stdev

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        landmarks = landmarks + torch.empty(3).normal_(0, self.stdev)
        return dict(inputs, landmarks=landmarks)


class RandomScale(Transform):
    def __init__(self, limit: float | tuple[float, float] = 0.1, **kwargs: Any):
        super().__init__(**kwargs)
        if not isinstance(limit, Iterable):
            limit = (1 - limit, 1 + limit)
        self.limit = limit

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        landmarks = landmarks * torch.empty(3).uniform_(self.limit[0], self.limit[1])
        return dict(inputs, landmarks=landmarks)


class RandomShear(Transform):
    def __init__(self, limit: float = 0.1, **kwargs: Any):
        super().__init__(**kwargs)
        self.limit = limit

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        axis = np.random.choice(3)
        rest = list(set(range(3)) - {axis})

        S = torch.eye(3)
        S[rest, axis] = torch.empty(2).uniform_(-self.limit, self.limit)
        return dict(inputs, landmarks=torch.einsum("ij,bni->bnj", S, landmarks))


class RandomInterpolatedRotation(Transform):
    def __init__(
        self,
        center_stdev: float = 0.5,
        angle_limit: float = np.pi / 4,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.center_stdev = center_stdev
        self.angle_limit = angle_limit

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        # In this rotation augmentation, the rotation center point and its rotation
        # angles will be contiguously changed throughout the timeline.
        offset = torch.lerp(
            torch.empty(3).normal_(0, self.center_stdev)[None, :],
            torch.empty(3).normal_(0, self.center_stdev)[None, :],
            torch.linspace(0, 1, landmarks.size(0))[:, None],
        )
        rotvec = torch.lerp(
            torch.empty(3).uniform_(-self.angle_limit, self.angle_limit)[None, :],
            torch.empty(3).uniform_(-self.angle_limit, self.angle_limit)[None, :],
            torch.linspace(0, 1, landmarks.size(0))[:, None],
        )

        R = Rotation.from_rotvec(rotvec.numpy()).as_matrix()
        R = torch.as_tensor(R, dtype=torch.float32)

        # To rotate the landmarks at a random point while maintaining original zero
        # point, we begin with subtracting the rotation center. Subsequently, a random
        # rotation is applied and the landmarks are then restored by adding the center.
        landmarks = landmarks - offset[:, None, :]
        landmarks = torch.einsum("bij,bni->bnj", R, landmarks) + offset[:, None, :]
        return dict(inputs, landmarks=landmarks)


class FrameBlockMask(Transform):
    def __init__(self, ratio: float = 0.1, block_size: int = 3, **kwargs: Any):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.block_size = block_size

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        # Because frames around the anchors will be masked as well, the probability
        # should be divided by the block size to maintain the masking ratio.
        mask = ~landmarks.isnan().all(2).all(1)
        mask = mask & (torch.rand(landmarks.size(0)) < self.ratio / self.block_size)
        mask = F.max_pool1d(mask[None].float(), self.block_size, 1, 1).squeeze(0).bool()

        landmarks = landmarks.masked_fill(mask[:, None, None], torch.nan)
        return dict(inputs, landmarks=landmarks)


class FrameNoise(Transform):
    def __init__(self, ratio: float = 0.1, noise_stdev: float = 1.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.noise_stdev = noise_stdev

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        mask = ~landmarks.isnan().all(2).all(1)
        mask = mask & (torch.rand(landmarks.size(0)) < self.ratio)
        noise = landmarks.clone().normal_(0, self.noise_stdev)
        return dict(inputs, landmarks=landmarks.where(~mask[:, None, None], noise))


class FeatureMask(Transform):
    def __init__(self, ratio: float = 0.1, **kwargs: Any):
        super().__init__(**kwargs)
        self.ratio = ratio

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        mask = torch.rand(landmarks.size(1)) < self.ratio
        landmarks = landmarks.masked_fill(mask[None, :, None], torch.nan)
        return dict(inputs, landmarks=landmarks)


def create_transform(augmentation: str, max_length: int) -> Transform:
    if augmentation == "valid":
        return Sequential(
            Normalize(),
            CenterCrop(max_length),
            Pad(max_length),
        )
    
    elif augmentation == "train":
        return Sequential(
            Normalize(),
            RandomResample(limit=0.3, p=0.5),
            RandomCrop(max_length),
            HorizontalFlip(p=0.5),
            # TimeFlip(p=0.5),
            FrameBlockMask(ratio=0.1, block_size=3, p=0.25),
            FrameNoise(ratio=0.1, noise_stdev=0.3, p=0.25),
            FeatureMask(ratio=0.1, p=0.1),
            RandomInterpolatedRotation(0.2, np.pi / 4, p=0.5),
            RandomShear(limit=0.2),
            RandomScale(limit=0.2),
            RandomShift(stdev=0.1),
            Pad(max_length),
        )