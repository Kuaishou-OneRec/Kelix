import unittest
import wids
import json
from torch.utils.data import DataLoader
from recovlm.data.datasets import BlendedWebDataset, BlendDatasetCkptManager
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
    if "this_worker_sample_index" == key:
      continue
    elif not deep_compare(dict1[key], dict2[key]):
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
    self.model_path = "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct"
  
  def _build_dataset(self, rank=0, world_size=1, num_workers=1, sample_buffer_size=2000):
    source_dataset = [wids.ShardListDataset(source) for source in self.data_sources]
    blend_dataset = BlendedWebDataset(source_dataset,
                       self.data_weight, 
                       rank=rank, world_size=world_size, num_workers=num_workers,
                       sample_buffer_size = sample_buffer_size,
                       random_seed=1024)
    return blend_dataset
  
  def _build_dataloader(self, dataset, num_workers=1, collator_func=None, batch_size=32):
    # build_dataloader
    if collator_func is None:
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
    else:
      collator = collator_func

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
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
    # init state_dict: state_dict[d_idx][w_idx]
    state_dict = {}
    
    # init datset
    blend_dataset = self._build_dataset(sample_buffer_size=1)
    data_iter1 = iter(blend_dataset)
    for _ in range(self.test_iter_num):
      data = next(data_iter1)
      d_idx, worker_id, chunk_idx, chunk_inner_idx, cur_samples_id = data["this_worker_sample_index"]
      state_dict.setdefault(d_idx, {})
      if worker_id not in state_dict[d_idx]:
        state_dict[d_idx][worker_id] = (chunk_idx, chunk_inner_idx, cur_samples_id)
      else:
        _, _, s = state_dict[d_idx][worker_id]
        if s < cur_samples_id:
          state_dict[d_idx][worker_id] = (chunk_idx, chunk_inner_idx, cur_samples_id)
    
    # reload checkpoint
    blend_dataset_reload = self._build_dataset(sample_buffer_size=1)
    blend_dataset_reload.set_state(state_dict)

    data_iter2 = iter(blend_dataset_reload)
    data2 = next(data_iter2)

    for _ in range(self.test_iter_num):
      data1 = next(data_iter1)
      data2 = next(data_iter2)
      assert blend_dataset_reload.dataset_states == blend_dataset.dataset_states
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
  
  def testDataloaderWithSave(self):
    blend_dataset = self._build_dataset(num_workers=1, sample_buffer_size=1)
    collator =  ImageTextPackingCollator(
        processor = AutoProcessor.from_pretrained(self.model_path),
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

    def dataset_collator(samples, collator):
      state_list = []
      for s in samples:
        state_list.append(s["this_worker_sample_index"])
      batch_state = BlendDatasetCkptManager.merge_all_state(state_list)
      batch_data = collator(samples)
      return batch_data, batch_state
  
    dataset_ckpt_manager = BlendDatasetCkptManager("/tmp/dataset_ckpt_path",
                                                   1, 1,
                                                   len(self.data_sources), 1)
    dataloader = self._build_dataloader(blend_dataset,
                                        num_workers=1,
                                        collator_func=lambda samples: dataset_collator(samples, collator), batch_size=1)
    # iter dataset
    data_iter = iter(dataloader)
    for step in range(80):
      _, state = next(data_iter)
      dataset_ckpt_manager.update_step(step, state)
      if step % 50 == 0 and step != 0:
        dataset_ckpt_manager.save_ckpt()
        break
    ######## reload dataset ########
    dataset_ckpt_manager = BlendDatasetCkptManager("/tmp/dataset_ckpt_path",
                                                   1, 1,
                                                   len(self.data_sources), 1)
    blend_dataset_reload = self._build_dataset(num_workers=1, sample_buffer_size=1)
    state_dict = dataset_ckpt_manager.load_latest_ckpt()
    print(state_dict)

    blend_dataset_reload.set_state(state_dict)
    print("=========")
    print([[list(n) for n in node] for node in blend_dataset.dataset_states])
    print([[list(n) for n in node] for node in blend_dataset_reload.dataset_states])
    print("=========")

    dataloader_reload = self._build_dataloader(blend_dataset_reload, num_workers=1, batch_size=1)

    print("=========")
    print([[list(n) for n in node] for node in blend_dataset.dataset_states])
    print([[list(n) for n in node] for node in blend_dataset_reload.dataset_states])
    print("=========")

    data_iter_reload = iter(dataloader_reload)
    d = next(data_iter_reload)
    d = next(data_iter_reload)
    for i in range(10):
      batch_data, _ = next(data_iter)
      reload_batch_data = next(data_iter_reload)
      print("=========")
      print([[list(n) for n in node] for node in blend_dataset.dataset_states])
      print([[list(n) for n in node] for node in blend_dataset_reload.dataset_states])
      print("=========")
      # assert batch_data == reload_batch_data
      print(batch_data)
      print(reload_batch_data)
      break
  
if __name__ == "__main__":
  unittest.main()