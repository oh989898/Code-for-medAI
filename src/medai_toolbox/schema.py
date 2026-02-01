from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class NodeCategory(str, Enum):
    DATA = "data"
    MODEL = "model"
    TRAIN = "train"
    EVAL = "eval"


@dataclass
class NodeParam:
    name: str
    value: Any
    description: Optional[str] = None


@dataclass
class NodeSchema:
    name: str
    category: NodeCategory
    params: Dict[str, NodeParam] = field(default_factory=dict)
    code_template: Optional[str] = None


@dataclass
class PipelineEdge:
    source: str
    target: str


@dataclass
class PipelineConfig:
    name: str = "medai_pipeline"
    nodes: List[NodeSchema] = field(default_factory=list)
    edges: List[PipelineEdge] = field(default_factory=list)
    output_dir: str = "generated"

    def get_node(self, category: NodeCategory) -> NodeSchema:
        for node in self.nodes:
            if node.category == category:
                return node
        raise ValueError(f"Missing required node category: {category}")
