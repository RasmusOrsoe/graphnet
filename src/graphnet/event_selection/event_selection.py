from abc import ABC, abstractmethod
from typing import List
import sqlite3
import pandas as pd
from graphnet.utilities.logging import LoggerMixin


class EventSelection(ABC, LoggerMixin):
    """Abstract Event Selection Class"""

    def __init__(self, levels):
        self._levels = levels

    def run(self, data_source, selection=None, index_column: str = "event_no"):
        if selection is None:
            selection = self._get_all_events(data_source, index_column)
        for level in self._levels:
            level(data_source, index_column, selection)
        return
