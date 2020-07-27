from enum import Enum


class AttentionSimilarityEnum(Enum):
    INNER_PRODUCT = 'inner-product'
    CONCAT = 'concat'
    GENERAL = 'general'
