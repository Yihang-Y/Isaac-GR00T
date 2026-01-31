"""
Model registry bootstrap.

Importing the setup modules has intentional side effects: they register model config
classes to their corresponding pipelines in `MODEL_REGISTRY`.
"""

from .gr00t_n1d6.setup import Gr00tN1d6Pipeline
from .gr00t_n1d6_mem.setup import Gr00tN1d6MemPipeline
from .registry import MODEL_REGISTRY
