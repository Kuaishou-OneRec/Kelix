import os
import math
import json
from tqdm import tqdm
from glob import glob
import pyarrow as pa
import pyarrow.parquet as pq
from tools.data_helpers.utils import MPIBase
from torch.utils.data import IterableDataset
import webdataset as wds
import tarfile
import base64
from PIL import Image
import io
import bs4
from bs4 import BeautifulSoup as bs
from html import escape

def lcm(a: int, b: int):
    return a * b // (math.gcd(a, b))

class DistDataset(IterableDataset, MPIBase):
    def __init__(self):
        super().__init__()
    
class ParquetDataset(DistDataset):

    def __init__(self, path, columns = None, user='mpi'):
        super().__init__()
        self.path = path
        if columns is not None:
            columns = list(columns)
        self.columns = columns
        self.user = user
        if isinstance(path, str):
            self.shard_files = self.get_shard_files(path)
        else:
            self.shard_files = []
            for p in path:
                self.shard_files += self.get_shard_files(p)
                
    def get_shard_files(self, path: str):
        if self.rank == 0:
            if path.startswith("viewfs"):
                self.fs = pa.hdfs.connect(user=self.user)
                files = self.fs.ls(path)
                files = sorted([x for x in files if x.endswith("parquet")  or "part-" in x])
            elif path.startswith("/"):
                files = sorted(glob(os.path.join(path, "*.parquet")))
            num_files = len(files)
            lcm_num_files_world_size = lcm(num_files, self.world_size)
            num_single_file_shard = lcm_num_files_world_size // num_files
            shard_files = [
                # filename, shard_id, shard_size
                (fn, sid, num_single_file_shard)
                for fn in files
                for sid in range(num_single_file_shard)
            ]
        else:
            shard_files = None
        shard_files = self.comm.bcast(shard_files, root=0)
        shard_files = shard_files[self.rank::self.world_size]
        self.mpi_print("shard_files_len", len(shard_files))
        return shard_files    
    
    def __iter__(self):
        for fn, sid, shard_size in self.shard_files:
            try:
                df = pq.read_table(fn, columns=self.columns).to_pandas()
            except Exception as e:
                print(f"msy Error reading file {fn}: {e}")
                continue
            if sid == 0:
                # self.mpi_print(f"====ParquetDataset====\nRead {fn}, total rows {len(df)}")
                pass
            df = df[df.index % shard_size == sid]
            for _, row in df.iterrows():
                row = row.to_dict()
                yield row

class JsonlDataset(DistDataset):
    
    def __init__(self, path):
        super().__init__()
        self.path = path
        with open(self.path, "r") as f:
            lines = f.readlines()
        shard_size = len(lines) // self.world_size
        self.lines = lines[self.rank * shard_size: (self.rank + 1) * shard_size]
    
    def __iter__(self):
        for l in self.lines:
            if l.strip() != '':
                yield json.loads(l)

class JsonDataset(DistDataset):

    def __init__(self, path):
        super().__init__()
        self.path = path
        with open(self.path, "r") as f:
            self.data = json.load(f)
        self.data = self.data[self.rank::self.world_size]
    
    def __iter__(self):
        for s in self.data:
            yield s


class WebDataset(DistDataset):

    def __init__(self, index_list):
        super().__init__()
        if self.rank == 0:
            items = []
            for index_fn in index_list:
                with open(index_fn, encoding="utf-8") as f:
                    index = json.loads(f.read())["shardlist"]
                    for item in index:
                        item['url'] = os.path.join(
                            os.path.dirname(index_fn),
                            item['url']
                        )
                        items.append(item)
        else:
            items = None
        self.items = self.comm.bcast(items, root=0)
        self.items = self.items[self.rank::self.world_size]
        self.total_samples = sum([x['nsamples'] for x in self.items])
        self.mpi_print(f"urls {len(self.items)}, total_samples {self.total_samples}")
    
    def __len__(self):
        return self.total_samples
    
    def __iter__(self):
        ds = wds.WebDataset(
            [x['url'] for x in self.items],
            handler=wds.warn_and_continue,
        )

        for s in ds:
            yield s

class TgzImageDataset(DistDataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.fns = []
        self.data_cnt = 0
        fn_list = [fn for fn in os.listdir(path) if fn.endswith("tar.gz")]
        for idx, fn in enumerate(fn_list):
            if idx % self.world_size == self.rank:
                self.fns.append(os.path.join(self.path, fn))
                with tarfile.open(os.path.join(self.path, fn), 'r:gz') as tar:
                    file_list = tar.getnames()
                    file_count = len(file_list)
                    self.data_cnt += file_count
        print(self.data_cnt, self.fns)
    
    def __len__(self):
        return self.data_cnt
    
    def __iter__(self):
        for fn in self.fns:
            with tarfile.open(fn, 'r:gz') as tar:
                for member in tar.getmembers():
                    file = tar.extractfile(member)
                    name = member.name
                    if file is not None:
                        file_bytes = file.read()
                        image_sample = (name, base64.b64encode(file_bytes).decode('ascii'))
                        yield image_sample

class VlmTextJsonl(DistDataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        with open(self.path, "r") as f:
            lines = f.readlines()
        shard_size = len(lines) // self.world_size
        self.lines = lines[self.rank * shard_size: (self.rank + 1) * shard_size]
    
    def __len__(self):
        return len(self.lines)
    
    def __iter__(self):
        for l in self.lines:
            if l.strip() != '':
                src = json.loads(l)
                if src["images"] is not None and len(src["images"]) > 0:
                    pass
                else:
                    yield src

class ImagesDataset(DistDataset):
    def __init__(self, path, img_max_size=0):
        super().__init__()
        self.img_path = []
        self.img_max_size = img_max_size

        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path) and "(" not in filename and ")" not in filename and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                self.img_path.append((file_path, filename))
        
        shard_size = len(self.img_path) // self.world_size
        self.img_path = self.img_path[self.rank * shard_size: (self.rank + 1) * shard_size]
    
    def __len__(self):
        return len(self.img_path)
    
    def resize_image(self, img):
        width, height = img.size
        if width > height:
            ratio = self.img_max_size / width
        else:
            ratio = self.img_max_size / height
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        img = img.resize((new_width, new_height))
        return img

    def read_image(self, img_path):
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                if self.img_max_size > 0 and max(img.size) > self.img_max_size:
                    img = self.resize_image(img)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()
                return img_bytes
        except Exception as e:
            print(f"Skipped {img_path}, err_msg={e}")
            return None

    def __iter__(self):
        for img_path, img_name in self.img_path:
            img_bytes = self.read_image(img_path)
            yield img_path, img_name, img_bytes

class PubTabNetDataset(DistDataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.jsonl_path = os.path.join(path, "PubTabNet_2.0.0.jsonl")
        
        # Read and shard the jsonl file
        with open(self.jsonl_path, "r") as f:
            lines = f.readlines()
        shard_size = len(lines) // self.world_size
        self.lines = lines[self.rank * shard_size: (self.rank + 1) * shard_size]
    
    def __len__(self):
        return len(self.lines)
    
    def format_html(self, img_data):
        html_code = img_data['html']['structure']['tokens'].copy()
        to_insert = [i for i, tag in enumerate(html_code) if tag in ('<td>', '>')]
        for i, cell in zip(to_insert[::-1], img_data['html']['cells'][::-1]):
            if cell['tokens']:
                cell = [escape(token) if len(token) == 1 else token for token in cell['tokens']]
                cell = ''.join(cell)
                html_code.insert(i + 1, cell)
        html_code = ''.join(html_code)
        html_code = '''<html>
                       <head>
                       <meta charset="UTF-8">
                       <style>
                       table, th, td {
                         border: 1px solid black;
                         font-size: 10px;
                       }
                       </style>
                       </head>
                       <body>
                       <table frame="hsides" rules="groups" width="100%%">
                         %s
                       </table>
                       </body>
                       </html>''' % html_code

        # prettify the html
        soup = bs(html_code)
        return soup.prettify()
    
    def __iter__(self):
        for line in self.lines:
            if line.strip() != '':
                data = json.loads(line)
                filename = data['filename']
                split = data['split']
                
                # Load image only when needed
                img_path = os.path.join(self.path, split, filename)
                if os.path.exists(img_path):
                    with open(img_path, 'rb') as f:
                        img_bytes = f.read()
                        data['image'] = base64.b64encode(img_bytes).decode('ascii')
                        data['html'] = self.format_html(data)
                        yield data
                        
if __name__ == "__main__":
    dataset = ImagesDataset('/llm_reco_ssd/zhangzixing/XFUND_img', 1080)
    for img_data in dataset:
        img_path, img_name, img_bytes = img_data
        print(img_path, len(img_bytes))