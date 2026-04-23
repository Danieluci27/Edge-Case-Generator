"""Dataset adapters for packaged local datasets."""

from edge_case_generator.collection.adapters.apps import APPSAdapter
from edge_case_generator.collection.adapters.codenet import CodeNetAdapter
from edge_case_generator.collection.adapters.codecontests import CodeContestsAdapter

__all__ = ["CodeNetAdapter", "CodeContestsAdapter", "APPSAdapter"]
