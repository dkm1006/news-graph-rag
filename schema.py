from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

import config
from utils import generate_short_uid


class ArticleChunkCategory(Enum):
    SUMMARY = 'summary'
    HEADLINE = 'headline'
    PARAGRAPH = 'paragraph'


@dataclass
class ArticleChunk:
    """An ArticleChunk is a text piece from an article with metadata"""
    text: str
    category: ArticleChunkCategory
    section: int
    position: int = 0
    embedding: np.ndarray = np.zeros(config.EMBEDDING_SIZE)
    uid: str = field(default_factory=lambda: generate_short_uid('Chunk', config.UID_LEN))

    def to_dict(self, serialize=False):
        result = self.__dict__
        if serialize:
            result['category'] = self.category.value
        
        return result


@dataclass(frozen=True)
class Entity:
    name: str
    label: str
