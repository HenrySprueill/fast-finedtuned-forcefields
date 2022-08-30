import torch
from torch.nn import functional as F
import os.path as osp
from ase.db import connect
from torch_geometric.data import InMemoryDataset, Data, extract_zip, download_url
from typing import Optional, Callable
import gdown
import sys
import tempfile
import os
import shutil

class TrainWaterDataSet(InMemoryDataset):
    #raw_url = "https://drive.google.com/uc?id=1ZQNOhJnz0k_UWxc-CIIkYwE2d5o230Ad&export=download"

    def __init__(self,
                 sample,
                 root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None
                 ):
        """

        Args:
            root: Directory where processed output will be saved
            transform: Transform to apply to data
            pre_transform: Pre-transform to apply to data
            pre_filter: Pre-filter to apply to data
        """
        self.atom_types = [1, 8]
        self.sample = sample
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # NB: database file
        return f'{self.sample}.db'

    @property
    def processed_file_names(self):
        return f'{self.sample}.pt'

    def download(self):
        """
        The base class will automatically look for a file that matches the raw_file_names property in a directory named 'raw'. If it doesn't find it, it will download the data using this method
        :return:
        """
        raw_file_path = osp.join(self.raw_dir, self.raw_file_names)
        if osp.exists(raw_file_path):
            print(f'Using existing file {self.raw_file_names}',
                  file=sys.stderr)
            return
        else:
            with tempfile.TemporaryFile() as fp:
                extract_zip(gdown.download(self.raw_url, fp), self.raw_dir)
            self.clean_up()

    def clean_up(self):
        """
        Remove the dataset that isn't used

        The datasets come bundled in a zip file. By default, both are downloaded and extracted. They can be fairly large, so this removes the dataset which isn't used.
        """
        db_to_remove = osp.join(self.raw_dir, 'even_split_subset_500k_test.db')
        if osp.exists(db_to_remove):
            os.remove(db_to_remove)

    def process(self):
        """
        Processes the raw data and saves it as a Torch Geometric data set

        The steps does all pre-processing required to put the data extracted from the database into graph 'format'. Several transforms are done on the data in order to generate the graph structures used by training.

        The processed dataset is automatically placed in a directory named processed, with the name of the processed file name property. If the processed file already exists in the correct directory, the processing step will be skipped.

        :return: Torch Geometric Dataset
        """
        # NB: coding for atom types
        types = {'H': 0, 'O': 1}

        data_list = []
        dbfile = osp.join(self.root, "raw", self.raw_file_names)
        assert osp.isfile(dbfile), f"Database file not found: {dbfile}"

        with connect(dbfile) as conn:
            center = True # hardcoding this for now to emulate SchnetPack
            for i, row in enumerate(conn.select()):
                if i % 50000 == 0:
                    print(f'atoms processed {i}')
                name = "energy"
                mol = row.toatoms()
                # get target (energy)
                #  The training target is the potential energy, which is stored in y
                y = torch.tensor(row.data[name], dtype=torch.float)
                if center:
                    pos = mol.get_positions() - mol.get_center_of_mass()
                else:
                    pos = mol.get_positions()
                pos = torch.tensor(pos, dtype=torch.float)
                type_idx = [types.get(i) for i in mol.get_chemical_symbols()]
                atomic_number = mol.get_atomic_numbers()
                z = torch.tensor(atomic_number, dtype=torch.long)
                x = F.one_hot(torch.tensor(type_idx, dtype=torch.long),
                              num_classes=len(self.atom_types))

                data = Data(x=x, z=z, pos=pos, y=y, name=name, idx=i)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                # The graph edge_attr and edge_indices are created when the transforms are applied
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
 
        torch.save(self.collate(data_list), self.processed_paths[0])

    
class TestWaterDataSet(TrainWaterDataSet):
    def __init__(self,
                 root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 ):

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        super().process()

    def download(self):
        super().download()

    def clean_up(self):
        db_to_remove = osp.join(self.raw_dir,
                                'even_split_subset_500k_train.db')
        if osp.exists(db_to_remove):
            os.remove(db_to_remove)

    @property
    def raw_file_names(self):
        # NB: database file
        return 'even_split_subset_500k_test.db'

    @property
    def processed_file_names(self):
        return 'even_split_subset_500k_test.pt'


class AllWaterDataSet(TrainWaterDataSet):

    raw_url = "https://drive.google.com/file/d/1UI_TzdkESvq81uscAe9Fs3KpQjGPaJQ2&export=download"

    def __init__(self,
                 root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 ):

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        super().process()

    @property
    def raw_file_names(self):
        # NB: database file
        return 'W3-25_geoms_all.db'

    @property
    def processed_file_names(self):
        return 'W3-25_geoms_all.pt'
    
    def download(self):
        """
        The base class will automatically look for a file that matches the raw_file_names property in a directory named 'raw'. If it doesn't find it, it will download the data using this method
        :return:
        """
        
        raw_file_path = osp.join(self.raw_dir, self.raw_file_names)
        if osp.exists(raw_file_path):
            print(f'Using existing file {self.raw_file_names}',
                  file=sys.stderr)
            return
        else:
            with tempfile.TemporaryFile() as fp:
                extract_zip(gdown.download(self.raw_url, fp), self.raw_dir)

class DebugWaterDataSet(TrainWaterDataSet):
    raw_url =  "https://graphcore-my.sharepoint.com/:u:/g/personal/mikek_graphcore_ai/EdshP0wspqBOi-iD7WlboIwB0UIyUA7sF0YrFgtvSZqH-g=download"

    def __init__(self,
                 root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 ):

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        super().process()

    def download(self):
        raw_file_path = osp.join(self.raw_dir, self.raw_file_names)
        if osp.exists(raw_file_path):
            print(f'Using existing file {self.raw_file_names}',
                  file=sys.stderr)
            return
        else:
            if not os.path.isdir(self.raw_dir):
                os.mkdir(self.raw_dir)
            src_file = osp.join('/localdata/mikek/work/pnnl_water_work/pnnl_sandbox/schnet/data/water_all_packed_k28/raw', self.raw_file_names)
            shutil.copy(src_file, raw_file_path)


    @property
    def raw_file_names(self):
        return 'small_water_sample.db'

    @property
    def processed_file_names(self):
        return 'small_water_sample.pt'
