from typing import List, Optional, Union
import pandas as pd
import sqlite3
import numpy as np
from graphnet.data.dataset import Dataset, ColumnMissingException


class SQLiteDataset(Dataset):
    """Pytorch dataset for reading from SQLite."""

    # Implementing abstract method(s)
    def _init(self):
        # Check(s)
        if isinstance(self._path, list):
            self._database_list = self._path
            self._all_connections_established = False
            self._all_connections = []
        else:
            self._database_list = None
            assert isinstance(self._path, str)
            assert self._path.endswith(
                ".db"
            ), f"Format of input file `{self._path}` is not supported."

        if self._database_list is not None:
            self._current_database = None

        # Set custom member variable(s)
        self._features_string = ", ".join(self._features)
        self._truth_string = ", ".join(self._truth)
        if self._node_truth:
            self._node_truth_string = ", ".join(self._node_truth)

        self._conn = None  # Handle for sqlite3.connection

    def _post_init(self):
        self._close_connection()

    def _query_table(
        self,
        table: str,
        columns: Union[List[str], str],
        index: int,
        selection: Optional[str] = None,
    ):
        """Query table at a specific index, optionally with some selection."""

        max_pulses = 300

        # Check(s)
        if isinstance(columns, list):
            n_features = len(columns)
            columns = ", ".join(columns)

        if not selection:  # I.e., `None` or `""`
            selection = "1=1"  # Identically true, to select all

        if self._database_list is None:
            index = self._indices[index]
        else:
            index = self._indices[index][0]

        # Query table
        self._establish_connection(index)
        try:
            if (
                self._add_inactive_sensors
                and "dom_x" in columns
                and n_features > 1
            ):  # last condition is to check if this is a pulsemap query
                active_query = f"select (CAST(dom_x AS str) || '_' || CAST(dom_y AS str) || '_' || CAST(dom_z AS str)) as UID, {columns} from {table} where {self._index_column} = {index} and {selection}"
                active_result = self._conn.execute(active_query).fetchall()
                if len(columns.split(", ")) > 1:
                    columns = ", ".join(
                        columns.split(", ")[1:]
                    )  # event_no not in geometry table
                query = f"select {columns} from {self._geometry_table} where UID not in {str(tuple(np.array(active_result)[:,0]))} limit {max_pulses}"
                inactive_result = self._conn.execute(query).fetchall()
                active_result = np.asarray(active_result)[
                    :, 2:
                ].tolist()  # drops UID column & event_no

                result = []
                if len(active_result) >= max_pulses:
                    result = active_result
                else:
                    result.extend(active_result)
                    result.extend(
                        inactive_result[0 : (max_pulses - len(active_result))]
                    )
                result = (
                    np.concatenate(
                        [np.repeat(index, len(result)).reshape(-1, 1), result],
                        axis=1,
                    )
                    .astype("float64")
                    .tolist()
                )
            else:
                result = self._conn.execute(
                    f"SELECT {columns} FROM {table} WHERE "
                    f"{self._index_column} = {index} and {selection}"
                ).fetchall()
        except sqlite3.OperationalError as e:
            if "no such column" in str(e):
                raise ColumnMissingException(str(e))
            else:
                raise e
        return result

    def _get_all_indices(self):
        self._establish_connection(0)
        indices = pd.read_sql_query(
            f"SELECT {self._index_column} FROM {self._truth_table}", self._conn
        )
        self._close_connection()
        return indices.values.ravel().tolist()

    # Customer, internal method(s)
    def _establish_connection(self, i):
        """Make sure that a sqlite3 connection is open."""
        if self._database_list is None:
            if self._conn is None:
                self._conn = sqlite3.connect(self._path)
        else:
            if self._conn is None:
                if self._all_connections_established is False:
                    self._all_connections = []
                    for database in self._database_list:
                        con = sqlite3.connect(database)
                        self._all_connections.append(con)
                    self._all_connections_established = True
                self._conn = self._all_connections[self._indices[i][1]]
            if self._indices[i][1] != self._current_database:
                self._conn = self._all_connections[self._indices[i][1]]
                self._current_database = self._indices[i][1]
        return self

    def _close_connection(self):
        """Make sure that no sqlite3 connection is open.

        This is necessary to calls this before passing to `torch.DataLoader`
        such that the dataset replica on each worker is required to create its
        own connection (thereby avoiding `sqlite3.DatabaseError: database disk
        image is malformed` errors due to inability to use sqlite3 connection
        accross processes.
        """
        if self._conn is not None:
            self._conn.close()
            del self._conn
            self._conn = None
        if self._database_list is not None:
            if self._all_connections_established:
                for con in self._all_connections:
                    con.close()
                del self._all_connections
                self._all_connections_established = False
                self._conn = None
        return self
