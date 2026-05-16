"""Compatibility imports for project pipelines.

New code should import from baseline_pipeline or bert_pipeline directly.
"""

from baseline_pipeline import BaselinePipeline
from bert_pipeline import BERTPipeline

__all__ = ["BaselinePipeline", "BERTPipeline"]
