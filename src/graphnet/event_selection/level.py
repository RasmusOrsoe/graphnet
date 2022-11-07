from abc import ABC, abstractmethod
import sqlite3
import pandas as pd
from typing import Any, List, Optional, Tuple, Union

from graphnet.data.dataset import ColumnMissingException
from graphnet.utilities.logging import LoggerMixin
from graphnet.data.sqlite.sqlite_utilities import (
    create_table,
    attach_index,
    save_to_sql,
)


class Level(ABC, LoggerMixin):
    """Abstract Event Selection Class"""

    @property
    def name(self):
        self.name

    @property
    def stages(self):
        return self._stages

    def __call__(self, data_source: str, index_column: str, selection):
        """Application Specific Method that runs the event selection."""
        if selection is None:
            selection = self._get_all_events(data_source, index_column)
            self._process(data_source, index_column, selection)
        return

    @abstractmethod
    def _process(self, data_source, index_column, selection):
        """Application Specific Method that determines if"""

    @abstractmethod
    def query(self):
        """Back-end specific method for loading data"""

    @abstractmethod
    def save_method(self):
        """Application Specific Method that runs the event selection."""

    @abstractmethod
    def _get_all_events(self, data_source, index_column):
        """Back-end specific method that retrieves all event no's in the data source"""


class SQLiteLevel(Level):
    """Abstract Event Selection Class"""

    def query(
        self,
        table: str,
        columns: Union[List[str], str],
        event_index: int,
        selection: Optional[str] = None,
    ) -> List[Tuple[Any, ...]]:
        """Query table at a specific index, optionally with some selection."""
        # Check(s)
        if isinstance(columns, list):
            columns = ", ".join(columns)

        if not selection:  # I.e., `None` or `""`
            selection = "1=1"  # Identically true, to select all

        # Query table
        self._establish_connection()
        try:
            assert self._conn
            if event_index is not None:
                result = self._conn.execute(
                    f"SELECT {columns} FROM {table} WHERE "
                    f"{self._index_column} = {event_index} and {selection}"
                ).fetchall()
            else:
                result = self._conn.execute(
                    f"SELECT {columns} FROM {table} WHERE {selection}"
                ).fetchall()
        except sqlite3.OperationalError as e:
            if "no such column" in str(e):
                raise ColumnMissingException(str(e))
            else:
                raise e
        return result

    def _establish_connection(self):
        """Make sure that a sqlite3 connection is open."""
        self._conn = sqlite3.connect(self._database)
        return

    def save_method(
        self,
        database: str,
        df: pd.DataFrame,
        is_pulse_map: bool,
        table_name: str,
    ):
        create_table(
            df=df,
            table_name=table_name,
            database_path=database,
            is_pulse_map=is_pulse_map,
        )
        if is_pulse_map:
            attach_index(database=database, table_name=table_name)
        save_to_sql(df=df, table_name=table_name, database=database)

    def _get_all_events(
        self, data_source: str, index_column: str
    ) -> pd.DataFrame:
        """Extract all event numbers from sqlite database"""
        assert data_source.endswith(".db")
        with sqlite3.connect(data_source) as con:
            query = f"SELECT {index_column} FROM {self._truth_table}"
            event_nos = pd.read_sql(query, con)
        return event_nos
