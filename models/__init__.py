"""
Neural networks: Transformer backbone (HF), Value (critic) head.
"""

from .backbone import TransformerBackbone
from .critic import ValueHead

__all__ = ["TransformerBackbone", "ValueHead"]
