"""CALE: collaborative adjustment based on lacunarity and entropy."""
from .roi_head import ConvFCCALEBBoxHead, Shared2FCCALEBBoxHead
from .atss_head import ATSSCALEHead

__all__ = ['ConvFCCALEBBoxHead', 'Shared2FCCALEBBoxHead', 'ATSSCALEHead']
