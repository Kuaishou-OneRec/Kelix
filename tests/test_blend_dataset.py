import unittest
import wids
import json
from recovlm.data.datasets import BlendedWebDataset

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
  def testBlendDataset(self):
    iter_num = 100
    sources = [
      "/llm_reco_ssd/luoxinchen/dataset/datacomp/large/index.json",
      "/llm_reco_ssd/luoxinchen/dataset/coyo-700m-webdataset/coyo-700m-index.json"
    ]

    dataset_weight = [0.8, 1.2]
    datasets1 = [wids.ShardListDataset(source) for source in sources]
    blend_dataset1 = BlendedWebDataset(datasets1,
                       dataset_weight, 
                       rank=0, world_size=1, num_workers=1, 
                       random_seed=1024)

    datasets2 = [wids.ShardListDataset(source) for source in sources]
    blend_dataset2 = BlendedWebDataset(datasets2,
                       dataset_weight, 
                       rank=0, world_size=1, num_workers=1, 
                       random_seed=1024)


    assert blend_dataset1.dataset_range_idx == blend_dataset2.dataset_range_idx
    assert blend_dataset1.dataset_states == blend_dataset2.dataset_states

    data_iter1 = iter(blend_dataset1)
    data_iter2 = iter(blend_dataset2)

    for _ in range(iter_num):
      data1 = next(data_iter1)
      data2 = next(data_iter2)
      assert deep_compare(data1, data2)

if __name__ == "__main__":
  unittest.main()