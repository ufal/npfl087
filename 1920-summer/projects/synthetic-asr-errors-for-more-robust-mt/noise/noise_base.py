from abc import abstractmethod


class NoiseGenerator:
    def __init__(self):
        pass

    @abstractmethod
    def generate_pronunciation(self, words_alpha_pos):
        pass

    @abstractmethod
    def generate_noise(self, list_sampling_nodes):
        pass

    @abstractmethod
    def get_name(self):
        pass
