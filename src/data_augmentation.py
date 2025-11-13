import numpy as np
from abc import ABC, abstractmethod

class AugmentationInterface(ABC):
    @abstractmethod
    def logic(self, x):
        pass

    def __call__(self, x):
        if np.random.rand() < self.probability:
            x = self.logic(x)
        return x


class FrequencyMasking(AugmentationInterface):
    def __init__(self, probability=0.5, max_width=5):
        self.probability = probability
        self.max_width = max_width
    
    def logic(self, data):
        data = data.copy()
        num_freqs = data.shape[-1]
        width = np.random.randint(1, self.max_width)
        f0 = np.random.randint(0, num_freqs - width)
        data[..., f0:f0+width] = 0
        return data

class TimeMasking(AugmentationInterface):
    def __init__(self, probability=0.5, max_width=10):
        self.probability = probability
        self.max_width = max_width

    def logic(self, data):
        data = data.copy()
        num_frames = data.shape[-2]
        width = np.random.randint(1, self.max_width)
        t0 = np.random.randint(0, num_frames - width)
        data[..., t0:t0+width, :] = 0
        return data
    
class AddNoise(AugmentationInterface):
    def __init__(self, probability=0.5, noise_level=0.01):
        self.probability = probability
        self.noise_level = noise_level

    def logic(self, data):
        data = data.copy()
        noise = np.random.randn(*data.shape) * self.noise_level
        return data + noise

class FrequencyShift(AugmentationInterface):
    def __init__(self, probability=0.5, shift_max=2):
        self.probablility = probability
        self.shift_max = shift_max

    def logic(self, data):
        data = data.copy()
        data = np.roll(data, np.random.randint(-self.shift_max, self.shift_max+1), axis=-1)
        return data

class AmplitudeJitter(AugmentationInterface):
    def __init__(self, probability=0.5, scale_range=(0.9, 1.1)):
        self.probability = probability
        self.scale_range = scale_range

    def logic(self, data):
        data = data.copy()
        scale = np.random.uniform(*self.scale_range)
        return data * scale

class SpecAugment(AugmentationInterface):    
    def __init__(self, probability=0.5, freq_mask_width=8, time_mask_width=12):
        self.probability = probability
        self.freq_masking = FrequencyMasking(probability=1.0, max_width=freq_mask_width)
        self.time_masking = TimeMasking(probability=1.0, max_width=time_mask_width)

    def logic(self, data):
        data = self.freq_masking.logic(data)
        data = self.time_masking.logic(data)
        return data
