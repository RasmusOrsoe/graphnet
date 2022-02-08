import pandas as pd
import numpy as np
import sqlite3
import torch
from torch_geometric.data import Data
import time

class SQLiteDataset(torch.utils.data.Dataset):
    """Pytorch dataset for reading from SQLite.
    """
    def __init__(self, database, pulsemap_table, features, truth, index_column='event_no', truth_table='truth', selection=None, dtype=torch.float32):

        # Check(s)
        if isinstance(database, list):
            self._database_list = database
            self._all_connections_established = False
            self._all_connections = []
        else:
            self._database_list = None
            assert isinstance(database, str)
            assert database.endswith('.db')

        assert isinstance(features, (list, tuple))
        assert isinstance(truth, (list, tuple))

        self._database = database
        self._pulsemap_table = pulsemap_table
        self._features = [index_column] + features
        self._truth = [index_column] + truth
        self._index_column = index_column
        self._truth_table = truth_table
        self._dtype = dtype

        self._features_string = ', '.join(self._features)
        self._truth_string = ', '.join(self._truth)
        if (self._database_list != None):
            self._current_database = None
        self._conn = None  # Handle for sqlite3.connection

        if selection is None:
            self._indices = self._get_all_indices()
        else:
            self._indices = selection
        self.close_connection()


    def __len__(self):
        return len(self._indices)

    def __getitem__(self, i):
        self.establish_connection(i)
        features, truth = self._query_database(i)
        graph = self._create_graph(features, truth)
        return graph

    def _get_all_indices(self):
        self.establish_connection(0)
        indices = pd.read_sql_query(f"SELECT {self._index_column} FROM {self._truth_table}", self._conn)
        return indices.values.ravel().tolist()

    def _query_database(self, i):
        """Query SQLite database for event feature and truth information.

        Args:
            i (int): Sequentially numbered index (i.e. in [0,len(self))) of the
                event to query.

        Returns:
            list: List of tuples, containing event features.
            list: List of tuples, containing truth information.
        """
        if self._database_list == None:
            index = self._indices[i]
        else:
            index = self._indices[i][0]


        features = self._conn.execute(
            "SELECT {} FROM {} WHERE {} = {}".format(
                self._features_string,
                self._pulsemap_table,
                self._index_column,
                index,
            )
        )

        truth = self._conn.execute(
            "SELECT {} FROM {} WHERE {} = {}".format(
                self._truth_string,
                self._truth_table,
                self._index_column,
                index,
            )
        )

        features = features.fetchall()
        truth = truth.fetchall()

        return features, truth

    def _create_graph(self, features, truth):
        """Create Pytorch Data (i.e.graph) object.

        No preprocessing is performed at this stage, just as no node adjancency
        is imposed. This means that the `edge_attr` and `edge_weight` attributes
        are not set.

        Args:
            features (list): List of tuples, containing event features.
            truth (list): List of tuples, containing truth information.

        Returns:
            torch.Data: Graph object.
        """
        # Convert nested list to simple dict
        truth_dict = {key: truth[0][ix] for ix, key in enumerate(self._truth)}
        assert len(truth) == 1

        # Unpack common variables
        abs_pid = abs(truth_dict['pid'])
        sim_type = truth_dict['sim_type']

        labels_dict = {
            'event_no': truth_dict['event_no'],
            'muon': int(abs_pid == 13),
            'muon_stopped': int(truth_dict.get('stopped_muon') == 1),
            'noise': int((abs_pid == 1) & (sim_type != 'data')),
            'neutrino': int((abs_pid != 13 ) & (abs_pid != 1 )),  # `abs_pid in [12,14,16]`?
            'v_e': int(abs_pid == 12),
            'v_u': int(abs_pid == 14),
            'v_t': int(abs_pid == 16),
            'track': int((abs_pid == 14) & (truth_dict['interaction_type'] == 1)),
            'dbang': int(sim_type == 'dbang'),
            'corsika': int(abs_pid > 20)
        }

        # Catch cases with no reconstructed pulses
        if len(features):
            data = np.asarray(features)[:,1:]
        else:
            data = np.array([]).reshape((0, len(self._features) - 1))

        # Construct graph data object
        x = torch.tensor(data, dtype=self._dtype)
        n_pulses = torch.tensor(len(x), dtype=torch.int32)
        graph = Data(
            x=x,
            edge_index= None
        )
        graph.n_pulses = n_pulses

        # Write attributes, either target labels or truth info.
        for write_dict in [labels_dict, truth_dict]:
            for key, value in write_dict.items():
                try:
                    if key in labels_dict.keys():
                        graph[key] = torch.tensor(value, dtype = torch.int)
                    else:
                        graph[key] = torch.tensor(value)
                except TypeError:
                    # Cannot convert `value` to Tensor due to its data type, e.g. `str`.
                    pass
        
        graph['XYZ'] = torch.tensor(np.array([truth_dict['position_x'],truth_dict['position_y'],truth_dict['position_z']]), dtype = torch.float32).reshape(-1,3)

        return graph

    def establish_connection(self,i):
        """Make sure that a sqlite3 connection is open."""
        if self._database_list == None:
            if self._conn is None:
                self._conn = sqlite3.connect(self._database)
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

    def close_connection(self):
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
        if self._database_list != None:
            if self._all_connections_established == True:
                for con in self._all_connections:
                    con.close()
                del self._all_connections
                self._all_connections_established = False
                self._conn = None
        return self


class SQLiteDatasetWithMergedPulses(torch.utils.data.Dataset):
    """Pytorch dataset for reading from SQLite.
    """
    def __init__(self, database, pulsemap_table, features, truth, index_column='event_no', truth_table='truth', selection=None, dtype=torch.float32,  time_window = 10):

        # Check(s)
        if isinstance(database, list):
            self._database_list = database
            self._all_connections_established = False
            self._all_connections = []
        else:
            self._database_list = None
            assert isinstance(database, str)
            assert database.endswith('.db')

        assert isinstance(features, (list, tuple))
        assert isinstance(truth, (list, tuple))
        print('--------------------------------')
        print('WARNING: THIS DATASET CLASS MERGES SAME-PMT PULSES WITHIN %s nanoseconds'%time_window)
        print('--------------------------------')
        self._database = database
        self._pulsemap_table = pulsemap_table
        self._features = [index_column] + features
        self._truth = [index_column] + truth
        self._index_column = index_column
        self._truth_table = truth_table
        self._dtype = dtype
        self._time_window = time_window

        self._features_string = ', '.join(self._features)
        self._truth_string = ', '.join(self._truth)
        if (self._database_list != None):
            self._current_database = None
        self._conn = None  # Handle for sqlite3.connection

        if selection is None:
            self._indices = self._get_all_indices()
        else:
            self._indices = selection
        self.close_connection()


    def __len__(self):
        return len(self._indices)

    def _merge_pulses(self, features):
        subset = ['dom_x','dom_y','dom_z']
        df = pd.DataFrame(data = features, columns = self._features).sort_values('dom_time').reset_index(drop = True)
        dups = df.loc[df.duplicated(subset = subset, keep = False),:].drop_duplicates(subset = subset)
        if len(dups) > 1:
            df_merged = df.copy()
            dropped = []
            for pulse in dups.index:
                pmt = df.loc[(df['dom_x'] == dups.loc[pulse, 'dom_x']) & (df['dom_y'] == dups.loc[pulse, 'dom_y']) & (df['dom_z'] == dups.loc[pulse, 'dom_z']), :]
                # IF ALL SAME-PMT CASES ARE WITHIN self.time_window NS        
                if (pmt['dom_time'][pmt.index[len(pmt)-1]] - pmt['dom_time'][pmt.index[0]]) < self._time_window:
                    df_merged.loc[pmt.index[0], 'charge'] = pmt['charge'].sum()
                    df_merged.loc[pmt.index[0], 'dom_time'] = np.ceil(pmt['dom_time'].mean())
                    df_merged = df_merged.drop(pmt.index[1:])
                    dropped.extend(pmt.index[1:])
                else:
                    for i in range(len(pmt)):
                        if pmt.index[i] in df_merged.index:
                            matches = pmt.loc[(abs(pmt['dom_time'] - pmt['dom_time'][pmt.index[i]]) < self._time_window),:] 
                            if len(matches) > 1 and  matches.index[0] not in dropped:
                                df_merged.loc[matches.index[0], 'charge'] = matches['charge'].sum()
                                df_merged.loc[matches.index[0], 'dom_time'] = np.ceil(matches['dom_time'].mean())
                                df_merged = df_merged.drop(matches.index[1:])
                                dropped.extend(matches.index[1:])
            return [tuple(r) for r in df_merged.to_numpy().tolist()]
        else:
            return features



    def __getitem__(self, i):
        self.establish_connection(i)
        features, truth = self._query_database(i)
        features = self._merge_pulses(features)
        graph = self._create_graph(features, truth)
        return graph

    def _get_all_indices(self):
        self.establish_connection(0)
        indices = pd.read_sql_query(f"SELECT {self._index_column} FROM {self._truth_table}", self._conn)
        return indices.values.ravel().tolist()

    def _query_database(self, i):
        """Query SQLite database for event feature and truth information.

        Args:
            i (int): Sequentially numbered index (i.e. in [0,len(self))) of the
                event to query.

        Returns:
            list: List of tuples, containing event features.
            list: List of tuples, containing truth information.
        """
        if self._database_list == None:
            index = self._indices[i]
        else:
            index = self._indices[i][0]


        features = self._conn.execute(
            "SELECT {} FROM {} WHERE {} = {}".format(
                self._features_string,
                self._pulsemap_table,
                self._index_column,
                index,
            )
        )

        truth = self._conn.execute(
            "SELECT {} FROM {} WHERE {} = {}".format(
                self._truth_string,
                self._truth_table,
                self._index_column,
                index,
            )
        )

        features = features.fetchall()
        truth = truth.fetchall()

        return features, truth

    def _create_graph(self, features, truth):
        """Create Pytorch Data (i.e.graph) object.

        No preprocessing is performed at this stage, just as no node adjancency
        is imposed. This means that the `edge_attr` and `edge_weight` attributes
        are not set.

        Args:
            features (list): List of tuples, containing event features.
            truth (list): List of tuples, containing truth information.

        Returns:
            torch.Data: Graph object.
        """
        # Convert nested list to simple dict
        truth_dict = {key: truth[0][ix] for ix, key in enumerate(self._truth)}
        assert len(truth) == 1

        # Unpack common variables
        abs_pid = abs(truth_dict['pid'])
        sim_type = truth_dict['sim_type']

        labels_dict = {
            'event_no': truth_dict['event_no'],
            'muon': int(abs_pid == 13),
            'muon_stopped': int(truth_dict.get('stopped_muon') == 1),
            'noise': int((abs_pid == 1) & (sim_type != 'data')),
            'neutrino': int((abs_pid != 13 ) & (abs_pid != 1 )),  # `abs_pid in [12,14,16]`?
            'v_e': int(abs_pid == 12),
            'v_u': int(abs_pid == 14),
            'v_t': int(abs_pid == 16),
            'track': int((abs_pid == 14) & (truth_dict['interaction_type'] == 1)),
            'dbang': int(sim_type == 'dbang'),
            'corsika': int(abs_pid > 20)
        }

        # Catch cases with no reconstructed pulses
        if len(features):
            data = np.asarray(features)[:,1:]
        else:
            data = np.array([]).reshape((0, len(self._features) - 1))

        # Construct graph data object
        x = torch.tensor(data, dtype=self._dtype)
        n_pulses = torch.tensor(len(x), dtype=torch.int32)
        graph = Data(
            x=x,
            edge_index= None
        )
        graph.n_pulses = n_pulses

        # Write attributes, either target labels or truth info.
        for write_dict in [labels_dict, truth_dict]:
            for key, value in write_dict.items():
                try:
                    if key in labels_dict.keys():
                        graph[key] = torch.tensor(value, dtype = torch.int)
                    else:
                        graph[key] = torch.tensor(value)
                except TypeError:
                    # Cannot convert `value` to Tensor due to its data type, e.g. `str`.
                    pass
        
        graph['XYZ'] = torch.tensor(np.array([truth_dict['position_x'],truth_dict['position_y'],truth_dict['position_z']]), dtype = torch.float32).reshape(-1,3)

        return graph

    def establish_connection(self,i):
        """Make sure that a sqlite3 connection is open."""
        if self._database_list == None:
            if self._conn is None:
                self._conn = sqlite3.connect(self._database)
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

    def close_connection(self):
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
        if self._database_list != None:
            if self._all_connections_established == True:
                for con in self._all_connections:
                    con.close()
                del self._all_connections
                self._all_connections_established = False
                self._conn = None
        return self


class SQLiteDatasetPerturbed(SQLiteDataset):
    """Pytorch dataset for reading from SQLite including a perturbation step to test the stability of a trained model.
    """
    def __init__(self, database, pulsemap_table, features, truth, perturbation_dict, index_column='event_no', truth_table='truth', selection=None, dtype=torch.float32, pertubate_pulsewise = True):

        assert isinstance(perturbation_dict, dict)
        assert len(set(perturbation_dict.keys())) == len(perturbation_dict.keys())
        self._perturbation_dict = perturbation_dict
        super().__init__(database, pulsemap_table, features, truth, index_column, truth_table, selection, dtype)
        self._perturbation_cols = [self._features.index(key) for key in self._perturbation_dict.keys()]
        self._pertube_pulsewise = pertubate_pulsewise
    def __getitem__(self, i):
        self.establish_connection(i)
        features, truth = self._query_database(i)
        perturbed_features = self._perturb_features(features)
        graph = self._create_graph(perturbed_features, truth)
        return graph

    def _perturb_features(self, features):
        features = np.array(features)
        if self._pertube_pulsewise:
            perturbed_features = np.random.normal(
                loc=features[:, self._perturbation_cols],
                scale=np.array(list(self._perturbation_dict.values()), dtype=np.float),
            )
            features[:, self._perturbation_cols] = perturbed_features
        else:
            # XY PERTUBATION
            unique_xy = np.unique(features[:,self._perturbation_cols[0:2]], axis = 0)
            #pertubed_features = deepcopy(features)
            for xy in unique_xy:
                pertubed_xy = np.random.normal(loc=xy, 
                                        scale=np.array(list(self._perturbation_dict.values())[0:2], 
                                        dtype=np.float)).reshape(1,-1)
                pertubed_z = np.random.normal(loc=0, 
                                        scale=np.array(list(self._perturbation_dict.values())[2], 
                                        dtype=np.float))
                idx = np.where((features[:,self._perturbation_cols[0]] == xy[0]) & (features[:,self._perturbation_cols[1]] == xy[1]))[0]
                #IN-PLACE XY
                features[np.ix_(idx,self._perturbation_cols[0:2])] = np.repeat(pertubed_xy,len(idx),axis = 0)
                #IN-PLACE Z
                features[idx,self._perturbation_cols[2]] = features[idx,self._perturbation_cols[2]] + pertubed_z
        return features