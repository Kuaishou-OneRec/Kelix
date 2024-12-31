import unittest
import wids
import json
from torch.utils.data import DataLoader
from recovlm.data.datasets import BlendedWebDataset
from recovlm.data.collators import ImageTextPackingCollator
from transformers import AutoTokenizer, AutoProcessor
import multiprocessing

def deep_compare(dict1, dict2):
  if type(dict1) is not dict or type(dict2) is not dict:
    return dict1 == dict2
  if len(dict1) != len(dict2):
    return False
  for key in dict1:
    if key not in dict2:
      return False
    if not deep_compare(dict1[key], dict2[key]):
      return False
  return True

class BlendDatasetTest(unittest.TestCase):
  def setUp(self):
    self.test_iter_num = 100
    self.data_sources = [
      "/llm_reco_ssd/luoxinchen/dataset/datacomp/large/index.json",
      "/llm_reco_ssd/luoxinchen/dataset/coyo-700m-webdataset/coyo-700m-index.json"
    ]
    self.data_weight = [0.8, 1.2]
    self.batch_size = 16
    self.model_path = "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct"
  
  def _build_dataset(self, rank=0, world_size=1, num_workers=1, sample_buffer_size=2000, state_file=""):
    source_dataset = [wids.ShardListDataset(source) for source in self.data_sources]
    blend_dataset = BlendedWebDataset(source_dataset,
                       self.data_weight, 
                       rank=rank, world_size=world_size, num_workers=num_workers,
                       sample_buffer_size = sample_buffer_size, state_file=state_file,
                       random_seed=1024)
    return blend_dataset
  
  def _build_dataloader(self, dataset, num_workers=1):
    # build_dataloader
    processor = AutoProcessor.from_pretrained(self.model_path)
    collator =  ImageTextPackingCollator(
      processor = processor,
      max_length = 1024,
      min_visual_tokens = 1,
      max_visual_tokens = 1024,
      spatial_merge_size = 2,
      image_token_id = 151655,
      video_token_id = 151656,
      vision_start_token_id = 151652,
      patch_size = 14,
      shrink_ratio = 0.9,
      max_retry = 5,
      multiple_of = 8
    )
    dataloader = DataLoader(
        dataset, batch_size=self.batch_size, num_workers=num_workers,
        collate_fn=collator
    )
    return dataloader

  def testBlendDataset(self):
    blend_dataset1 = self._build_dataset()
    blend_dataset2 = self._build_dataset()

    assert blend_dataset1.dataset_range_idx == blend_dataset2.dataset_range_idx
    assert blend_dataset1.dataset_states == blend_dataset2.dataset_states

    data_iter1 = iter(blend_dataset1)
    data_iter2 = iter(blend_dataset2)

    for _ in range(self.test_iter_num):
      data1 = next(data_iter1)
      data2 = next(data_iter2)
      assert deep_compare(data1, data2)

  def testBlendDatasetStates(self):
    # init state_dict: state_dict[self.rank][d_idx][w_idx]
    state_dict = []
    for i in range(1):
      state_dict.append([])
      for j in range(len(self.data_sources)):
        state_dict[i].append([])
        for k in range(1):
          state_dict[i][j].append(0)
    
    # init datset
    blend_dataset = self._build_dataset(sample_buffer_size=1)
    data_iter1 = iter(blend_dataset)
    for _ in range(self.test_iter_num):
      data = next(data_iter1)
      d_idx, worker_id, chunk_idx, chunk_inner_idx = data["this_worker_sample_index"]
      state_dict[0][d_idx][worker_id] = [chunk_idx, chunk_inner_idx]
    
    # dump checkpoint
    state_dump = json.dumps(state_dict)
    with open("/tmp/test.state.json", "w+") as fp:
      fp.write(state_dump)
    
    # reload checkpoint
    blend_dataset_reload = self._build_dataset(sample_buffer_size=1, state_file="/tmp/test.state.json")
    data_iter2 = iter(blend_dataset_reload)
    data2 = next(data_iter2)
    for _ in range(self.test_iter_num):
      data1 = next(data_iter1)
      data2 = next(data_iter2)
      assert deep_compare(data1, data2)
  
  def testDataloader(self):
    num_workers = 4
    blend_dataset = self._build_dataset(num_workers=num_workers)
    dataloader = self._build_dataloader(blend_dataset, num_workers=num_workers)

    # iter dataset
    data = iter(dataloader)
    for _ in range(self.test_iter_num):
      batch_data = next(data)
      data_id = id(batch_data) # for not opt
  
  def testMpiDataloader(self):
    def data_loader_test_func(rank, world_size):
      num_workers = 4
      blend_dataset = self._build_dataset(rank=rank, world_size=world_size, num_workers=num_workers)
      dataloader = self._build_dataloader(blend_dataset, num_workers=num_workers)

      # iter dataset
      data = iter(dataloader)
      for _ in range(self.test_iter_num):
        batch_data = next(data)
        data_id = id(batch_data) # for not opt
      print(f"[rank{rank}] dataloader finish")
    
    world_size = 8
    proc_list = []
    for rank in range(world_size):
      p = multiprocessing.Process(target=data_loader_test_func, args=(rank, world_size))
      p.start()
      proc_list.append(p)
    
    for p in proc_list:
      p.join()
  
  # def testMpiConsumeAll(self):
  #   def data_loader_test_func(rank, world_size):
  #     print(f"[rank{rank}] dataloader start test")
  #     num_workers = 8
  #     blend_dataset = self._build_dataset(rank=rank, world_size=world_size, num_workers=num_workers)
  #     dataloader = self._build_dataloader(blend_dataset, num_workers=num_workers)

  #     # iter dataset
  #     ans = 0
  #     for batch_data in dataloader:
  #       data_id = id(batch_data) # for not opt
  #       ans += 1
  #       if ans % 1000 == 0:
  #         print(f"[rank{rank}] load {ans} batchs")
  #     print(f"[rank{rank}] dataloader finish, total {ans} batchs")
    
  #   world_size = 8
  #   proc_list = []
  #   for rank in range(world_size):
  #     p = multiprocessing.Process(target=data_loader_test_func, args=(rank, world_size))
  #     p.start()
  #     proc_list.append(p)
    
  #   for p in proc_list:
  #     p.join()
  
if __name__ == "__main__":
  unittest.main()