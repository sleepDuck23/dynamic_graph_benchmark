from torch_geometric_temporal.dataset import METRLADatasetLoader, PemsBayDatasetLoader, TwitterTennisDatasetLoader, PedalMeDatasetLoader, MontevideoBusDatasetLoader
from pydgn.data.dataset import TemporalDatasetInterface
import numpy as np
from os.path import join, isfile, isdir
import pandas as pd
import torch
import kaggle
from tqdm import tqdm
from torch_geometric.data import Data, download_url, extract_zip
from torch_geometric.utils import from_scipy_sparse_matrix
import os
import pickle
import requests
import zipfile

# **** NODE PREDICTION ON SPATIO-TEMPORAL GRAPHS ****
class METRLADatasetInterface(TemporalDatasetInterface):
    """
    METR LA dataset.
    It contains spatio-temporal graphs for traffic forecasting.
    """

    def __init__(self, root, name='metr_la', num_timesteps_in=1, num_timesteps_out=1):

        self.root = root
        self.name = name
        self.num_timesteps_in = num_timesteps_in
        self.num_timesteps_out = num_timesteps_out

        path = join(self.root, self.name) + '.pt'
        if not isfile(path):
            self.dataset = METRLADatasetLoader(raw_data_dir = root)
            self.dataset = self.dataset.get_dataset(num_timesteps_in = self.num_timesteps_in,
                                                    num_timesteps_out = self.num_timesteps_out)

            self.dataset.features = [x.squeeze() for x in self.dataset.features]
            self.dataset.targets = [y.squeeze() for y in self.dataset.targets]
            torch.save(self.dataset, path)
        else:
            self.dataset = torch.load(path)


    @property
    def dim_node_features(self):
        return self.dataset.features[0].shape[1]

    @property
    def dim_edge_features(self):
        return 0

    @property
    def dim_target(self):
        return 1

    def get_mask(self, data):
        # in this case data is a Data object containing a snapshot of a single
        # graph sequence.
        # the task is node classification at each time step
        mask = np.ones((1,1))  #  time_steps x 1
        return mask

    def __len__(self):
        return len(self.dataset.features)

    def __getitem__(self, time_index):
        data = self.dataset.__getitem__(time_index)
        setattr(data, 'mask', self.get_mask(data))
        return data


class TrafficDatasetInterface(TemporalDatasetInterface):
    """
    Traffic dataset.
    It contains spatio-temporal graphs for traffic forecasting.
    """

    def __init__(self, root, name='traffic', num_timesteps_in=1, num_timesteps_out=1):
        
        if num_timesteps_in!=1 or num_timesteps_out!=1:
            raise NotImplementedError()

        self.root = root
        self.name = name
        self.num_timesteps_in = num_timesteps_in
        self.num_timesteps_out = num_timesteps_out

        path = join(self.root, self.name) + '.pt'
        if not isfile(path):
            self.load_data_()
            torch.save(self.dataset, path)
        else:
            self.dataset = torch.load(path)

    def load_data_(self):
        zip_file = download_url('https://github.com/chocolates/Predicting-Path-Failure-In-Time-Evolving-Graphs/raw/master/DATA/adj_list.pkl.zip', self.root)
        extract_zip(zip_file, self.root)
        os.unlink(zip_file)
        adj_path = os.path.join(self.root, 'adj_list.pkl')
        adj = pickle.load(open(adj_path, 'rb'))[0]
        edge_index, _ = from_scipy_sparse_matrix(adj)
        edge_index = edge_index.to(torch.long)

        zip_file = download_url('https://github.com/chocolates/Predicting-Path-Failure-In-Time-Evolving-Graphs/raw/master/DATA/input_feature.pkl.zip', self.root)
        extract_zip(zip_file, self.root)
        os.unlink(zip_file)
        ft_path = os.path.join(self.root, 'input_feature.pkl')
        ft = pickle.load(open(ft_path, 'rb'))
        
        self.dataset = []
        for i in range(len(ft)-1):
            self.dataset.append(Data(edge_index=edge_index,
                                     x = torch.from_numpy(ft[i]).float(),
                                     y = torch.from_numpy(ft[i+1]).float())
            )
                            
    @property
    def dim_node_features(self):
        return self.dataset[0].x.shape[1]

    @property
    def dim_edge_features(self):
        return 0

    @property
    def dim_target(self):
        return 2

    def get_mask(self, data):
        # in this case data is a Data object containing a snapshot of a single
        # graph sequence.
        # the task is node classification at each time step
        mask = np.ones((1,1))  #  time_steps x 1
        return mask

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, time_index):
        data = self.dataset[time_index]
        setattr(data, 'mask', self.get_mask(data))
        return data


class PemsBayDatasetInterface(METRLADatasetInterface):
    """
    Pems Bay dataset.
    It contains spatio-temporal graphs for traffic forecasting.
    """

    def __init__(self, root, name='pems_bay', num_timesteps_in=1, num_timesteps_out=1) :
        self.root = root
        self.name = name
        self.num_timesteps_in = num_timesteps_in
        self.num_timesteps_out = num_timesteps_out

        path = join(self.root, self.name) + '.pt'
        if not isfile(path):
            self.dataset = PemsBayDatasetLoader(raw_data_dir = root)
            self.dataset = self.dataset.get_dataset(num_timesteps_in = self.num_timesteps_in,
                                                    num_timesteps_out = self.num_timesteps_out)

            self.dataset.features = [x.squeeze() for x in self.dataset.features]
            self.dataset.targets = [y.squeeze() for y in self.dataset.targets]
            self.dataset.targets = [y.squeeze() for y in self.dataset.targets]
            torch.save(self.dataset, path)
        else:
            self.dataset = torch.load(path)

    @property
    def dim_target(self):
        # node regression: each time step is a tuple <velocity, volume>
        return 2


class PedalMeDatasetInterface(METRLADatasetInterface):
    """
    PedalMe dataset.
    It contains spatio-temporal graphs for traffic forecasting.
    """

    def __init__(self, root, name='pedalme', lags=4) :
        self.root = root
        self.name = name
        self.lags = lags

        path = join(self.root, self.name) + '.pt'
        if not isfile(path):
            self.dataset = PedalMeDatasetLoader()
            self.dataset = self.dataset.get_dataset(lags = self.lags)
            torch.save(self.dataset, path)
        else:
            self.dataset = torch.load(path)

    @property
    def dim_target(self):
        return 1


class MontevideoBusDatasetInterface(METRLADatasetInterface):
    """
    MontevideoBus dataset.
    It contains spatio-temporal graphs for inflow passenger forecasting.
    """

    def __init__(self, root, name='montevideo', lags=4) :
        self.root = root
        self.name = name
        self.lags = lags

        path = join(self.root, self.name) + '.pt'
        if not isfile(path):
            self.dataset = MontevideoBusDatasetLoader()
            self.dataset = self.dataset.get_dataset(lags = self.lags)
            torch.save(self.dataset, path)
        else:
            self.dataset = torch.load(path)

    @property
    def dim_target(self):
        return 1
    
# New class/dataset for the DL project (Citi Bike of New York 2017)
class CitiBikeDatasetInterface(TemporalDatasetInterface):
    """
    Citi Bike Dataset Interface for spatio-temporal graph node prediction.
    Combines monthly CSV files into a single dataset for easier processing.
    """

    def __init__(self, root, url, num_timesteps_in=1, num_timesteps_out=1):
        # Initialize the parent class
        super().__init__(root=root, num_timesteps_in=num_timesteps_in, num_timesteps_out=num_timesteps_out)

        self.root = root
        self.url = url
        self.num_timesteps_in = num_timesteps_in
        self.num_timesteps_out = num_timesteps_out

        # Ensure the dataset is downloaded and extracted
        self.data_path = self.download_and_extract()

        # Combine all monthly CSV files into a single CSV file
        self.combined_file = "./combined_citibike_data.csv"
        self.combine_csv_files(self.combined_file)

        # Load the combined CSV file and create temporal windows
        self.temporal_data = self.load_and_create_temporal_windows()

    def download_and_extract(self):
        # Ensure the root directory exists
        os.makedirs(self.root, exist_ok=True)

        zip_path = join(self.root, "2017-citibike-tripdata.zip")
        extract_path = join(self.root, "2017-citibike-tripdata")

        if not os.path.exists(zip_path):
            print("Downloading dataset...")
            response = requests.get(self.url)
            if response.status_code == 200:
                with open(zip_path, "wb") as f:
                    f.write(response.content)
                print("Dataset downloaded.")
            else:
                raise Exception(f"Failed to download dataset. Status code: {response.status_code}")

        # Extract the zip file if not already extracted
        if not os.path.exists(extract_path):
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
            print("Dataset extracted.")
        else:
            print("Dataset already extracted.")

        return extract_path

    def combine_csv_files(self, output_file):
        """
        Combines all monthly CSV files into a single CSV (optimized for low memory usage).
        """
        print("Combining monthly data into a single CSV (optimized for low memory)...")

        # Open the output file in write mode
        first_file = True  # To track whether to write the header
        for month_folder in sorted(os.listdir(self.data_path)):
            month_path = join(self.data_path, month_folder)
            if os.path.isdir(month_path):
                for file in os.listdir(month_path):
                    if file.endswith(".csv"):
                        file_path = join(month_path, file)
                        print(f"Processing {file_path} in chunks...")

                        # Read file in chunks
                        for chunk in pd.read_csv(file_path, chunksize=10000):  # Adjust chunksize as needed
                            # Normalize column names
                            chunk.columns = chunk.columns.str.strip().str.lower()

                            # Write header only for the first file
                            chunk.to_csv(output_file, mode='a', index=False, header=first_file)
                            first_file = False  # After the first chunk, don't write the header again

        print(f"All data combined into {output_file}.")

    def load_and_create_temporal_windows(self):
        """
        Loads the combined CSV file and converts the data into temporal windows.
        """
        print("Loading combined data and creating temporal windows...")

        # Read the combined file in chunks
        temporal_data = []
        for chunk in pd.read_csv(self.combined_file, chunksize=10000):  # Adjust chunksize if necessary
            # Parse datetime for sorting and consistency
            chunk['starttime'] = pd.to_datetime(chunk['starttime'])
            chunk.sort_values('starttime', inplace=True)

            # Normalize column names for consistent processing
            chunk.columns = chunk.columns.str.strip().str.lower()

            # Ensure data is sorted by 'starttime'
            chunk.reset_index(drop=True, inplace=True)

            # Create temporal windows
            temporal_data.extend(self._create_temporal_windows(chunk))

        return temporal_data

    def _create_temporal_windows(self, data):
        """
        Creates input-output temporal windows from the dataset.
        Each window contains `num_timesteps_in` for input and `num_timesteps_out` for output.
        """
        temporal_windows = []
        for i in range(len(data) - self.num_timesteps_in - self.num_timesteps_out + 1):
            # Input window
            x = data.iloc[i : i + self.num_timesteps_in]
            # Output window
            y = data.iloc[i + self.num_timesteps_in : i + self.num_timesteps_in + self.num_timesteps_out]
            temporal_windows.append((x, y))
        return temporal_windows

    def get_data(self):
        """
        Returns the temporal data as a list of input-output pairs.
        """
        return self.temporal_data



# **** NODE PREDICTION ON DISCRETE-DYNAMIC GRAPHS ****
class TwitterTennisDatasetInterface(TemporalDatasetInterface):
    """
    Twitter Tennis Dataset.
    It contains Twitter mention graphs related to major tennis tournaments from 2017.
    Each snapshot change with respect to edges and features.
    """

    def __init__(self, root, name, event_id='rg17', num_nodes=1000,
                 feature_mode='encoded', target_offset=1) :

        assert event_id in ['rg17', 'uo17'], f'event_id can be rg17 or uo17, not {event_id}'
        assert num_nodes <= 1000, f'num_nodes must be less or equal to 1000, not {num_nodes}'
        assert feature_mode in [None, 'diagonal', 'encoded'], f'feature_mode can be None, diagonal, or encoded. It can not be {feature_mode}'

        self.root = root
        self.name = name
        self.event_id = event_id
        self.num_nodes = num_nodes
        self.feature_mode = feature_mode
        self.target_offset = target_offset

        path = join(self.root, self.name) + '.pt'
        if not isfile(path):
            self.dataset = TwitterTennisDatasetLoader(
                event_id = self.event_id,
                N = self.num_nodes,
                feature_mode = self.feature_mode,
                target_offset = self.target_offset
            ).get_dataset()
            
            torch.save(self.dataset, path)
        else:
            self.dataset = torch.load(path)

    @property
    def dim_node_features(self):
        return self.dataset.features[0].shape[1]

    @property
    def dim_edge_features(self):
        return 1

    @property
    def dim_target(self):
        # node regression: each time step is a tuple
        return 1
    
    def get_mask(self, data):
        # in this case data is a Data object containing a snapshot of a single
        # graph sequence.
        # the task is node classification at each time step
        mask = np.ones((1,1))  #  time_steps x 1
        return mask

    def __len__(self):
        return len(self.dataset.features)

    def __getitem__(self, time_index):
        data = self.dataset.__getitem__(time_index)
        setattr(data, 'mask', self.get_mask(data))
        setattr(data, 'relation_type', data.edge_attr - 1)
        return data


class EllipticDatasetInterface(TemporalDatasetInterface):
    """
    Elliptic Dataset.
    The dataset maps Bitcoin transactions to real entities belonging to licit categories versus 
    illicit ones.
    Each snapshot change with respect to nodes, edges, and features.
    """

    def __init__(self, root, name='elliptic', fixed_nodes=True):
        self.root = root
        self.name = name
        self.fixed_nodes = fixed_nodes
        path = self._check_and_download()

        if not isfile(path + '.pt'):
            self.dataset = self._load_data(path)
        else:
            self.dataset = torch.load(path + '.pt')

    def _check_and_download(self):
        path = join(self.root, self.name)
        if not isdir(path):
            print('Downloading data...')
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files('ellipticco/elliptic-data-set', path=path, unzip=True) # <self.root>/<self.name>
        return path

    def _load_data(self, path):
        print('Loading data...')
        path_classes = join(path,'elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
        path_edgelist = join(path, 'elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
        path_features = join(path,'elliptic_bitcoin_dataset/elliptic_txs_features.csv')
 
        classes = pd.read_csv(path_classes, index_col = 'txId') # labels for the transactions i.e. 'unknown', '1', '2'
        edgelist = pd.read_csv(path_edgelist, index_col = 'txId1') # directed edges between transactions
        features = pd.read_csv(path_features, header = None, index_col = 0) # features of the transactions
        
        # Select only the labeled transactions
        labelled_classes = classes[classes['class'] != 'unknown']
        labelled_tx = set(list(labelled_classes.index))

        # Map node_id in the range [0, num_nodes] 
        nodes = features.index
        map_id = {j:i for i,j in enumerate(nodes)} # mapping nodes to indexes

        map_class = {'1':0, '2':1}

        # Compute Data object for each timestep
        data_list = []
        for timestep, df_group in tqdm(features.groupby(1)):
            # Keep only labelled nodes
            labelled_nodes = sorted([tx for tx in df_group.index if tx in labelled_tx])
            df_group = df_group.loc[labelled_nodes]

            # Get edge_index associated to the current timestep
            edge_index = edgelist.loc[edgelist.index.intersection(labelled_nodes).unique()]
            if self.fixed_nodes:
                # We consider only edges and features as dynamic, i.e., nodes are fixed on the temporal axis
                x = torch.tensor(features[range(2,167)].values, dtype = torch.float).contiguous()
                edge_index['txId1'] = edge_index.index.map(map_id)
                edge_index.txId2 = edge_index.txId2.map(map_id)
                node_mask = torch.zeros(x.shape[0])
                node_mask[[map_id[n] for n in labelled_nodes]] = 1
            else:
                # We consider nodes, edges and features as dynamic
                x = torch.tensor(df_group[range(2,167)].values, dtype = torch.float).contiguous()    
                map_id = {j:i for i,j in enumerate(labelled_nodes)}
                edge_index['txId1'] = edge_index.index.map(map_id)
                edge_index.txId2 = edge_index.txId2.map(map_id)
                node_mask = torch.ones(x.shape[0])

            edge_index = np.array(edge_index[['txId1', 'txId2']].values).T
            edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
            
            targets = [map_class[v] for v in classes.loc[labelled_nodes, 'class'].tolist()]
            y = torch.tensor(targets).contiguous()
            
            relation_type = torch.zeros(edge_index.shape[1])
            data = Data(
                edge_index = edge_index, 
                x = x,
                y = y,
                relation_type = relation_type, #torch.zeros(edge_index.shape[1]),
                node_mask = node_mask.bool()
            )
            data_list.append(data)

        torch.save(data_list, path + '.pt')
        return data_list

    '''
        for timestep in tqdm(range(49)):
            ft = features[features[1] == timestep+1]
            nodes_in_timestep = list(ft.index)
            labelled_nodes = [tx for tx in nodes_in_timestep if tx in labelled_tx]
            labelled_nodes = sorted(labelled_nodes)
            
            # Get edge_index associated to the current timestep
            edge_index = edgelist.loc[edgelist.index.intersection(labelled_nodes).unique()]

            if not self.fixed_nodes:
                x = torch.tensor(features.loc[labelled_nodes, range(2,167)].values, dtype = torch.float).contiguous()    
                map_id = {j:i for i,j in enumerate(labelled_nodes)}
                edge_index['txId1'] = edge_index.index.map(map_id)
                edge_index.txId2 = edge_index.txId2.map(map_id)
                
                node_mask = torch.zeros(x.shape[0])
                node_mask[[map_id[n] for n in labelled_nodes]] = 1
            else:
                x = torch.tensor(features[range(2,167)].values, dtype = torch.float).contiguous()
                edge_index['txId1'] = edge_index.index.map(map_id)
                edge_index.txId2 = edge_index.txId2.map(map_id)
                node_mask = torch.ones(x.shape[0])

            edge_index = np.array(edge_index[['txId1', 'txId2']].values).T
            edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()

            y = torch.tensor(classes.loc[labelled_nodes].map(map_class)).contiguous()

            data = Data(
                edge_index = edge_index, 
                x = x,
                y = y,
                node_mask = node_mask
            )
            data_list.append(data)

        torch.save(data_list, path + '.pt')
        return data_list
    '''

    def __getitem__(self, idx):
        data = self.dataset[idx]
        setattr(data, 'mask', self.get_mask(data))
        return data

    def get_mask(self, data):
        # We predict at each snapshot all the nodes
        mask = np.ones((1,1))  #  time_steps x 1
        return mask

    @property
    def dim_node_features(self):
        return self.dataset[0].x.shape[-1]

    @property
    def dim_edge_features(self):
        return 1

    @property
    def dim_target(self):
        # binary node classification
        return 1

    def __len__(self):
        return len(self.dataset)
