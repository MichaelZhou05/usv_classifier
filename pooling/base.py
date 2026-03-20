"""
Abstract base class for call pooling strategies.

Pooling transforms variable-length call sequences into fixed-size representations.
"""

from abc import ABC, abstractmethod

import numpy as np


class CallPooler(ABC):
    """
    Abstract base class for call pooling strategies.

    A pooler takes a variable number of call feature vectors and produces
    a fixed-size summary vector that represents the entire recording.
    """

    @abstractmethod
    def pool(self, call_features: np.ndarray) -> np.ndarray:
        """
        Pool multiple call feature vectors into a single summary vector.

        Args:
            call_features: (n_calls, n_features) array where each row is a call's
                          feature vector. n_calls can vary between recordings.

        Returns:
            Summary vector of shape (output_dim,). The output dimension is
            determined by the specific pooling strategy.

        Notes:
            - Must handle edge cases like empty arrays (n_calls=0)
            - Should be deterministic for reproducibility
        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """
        Return output dimension after pooling.

        This is the size of the vector returned by pool().
        """
        pass

    @property
    def name(self) -> str:
        """Return the name of this pooler (for logging)."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.name}(output_dim={self.output_dim})"
