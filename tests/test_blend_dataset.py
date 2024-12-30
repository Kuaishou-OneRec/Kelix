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

  def testDatasetStates(self):
    # TODO: error when sample_buffer_size!=1 !!
    iter_num = 100
    sources = [
      "/llm_reco_ssd/luoxinchen/dataset/datacomp/large/index.json",
      "/llm_reco_ssd/luoxinchen/dataset/coyo-700m-webdataset/coyo-700m-index.json"
    ]

    dataset_weight = [0.8, 1.2]
    datasets = [wids.ShardListDataset(source) for source in sources]
    blend_dataset = BlendedWebDataset(datasets,
                       dataset_weight, 
                       rank=0, world_size=1, num_workers=1, 
                       random_seed=1024, sample_buffer_size=1)


    # state_dict[self.rank][d_idx][w_idx]
    state_dict = []
    for i in range(1):
      state_dict.append([])
      for j in range(len(sources)):
        state_dict[i].append([])
        for k in range(1):
          state_dict[i][j].append(0)

    data_iter1 = iter(blend_dataset)
    for _ in range(iter_num):
      data = next(data_iter1)
      d_idx, worker_id, chunk_idx, chunk_inner_idx = data["this_worker_sample_index"]
      state_dict[0][d_idx][worker_id] = [chunk_idx, chunk_inner_idx]
    
    state_dump = json.dumps(state_dict)
    with open("/tmp/test.state.json", "w+") as fp:
      fp.write(state_dump)
    
    blend_dataset_reload = BlendedWebDataset(datasets,
                       dataset_weight, 
                       rank=0, world_size=1, num_workers=1, 
                       random_seed=1024, state_file="/tmp/test.state.json", sample_buffer_size=1)
    
    data_iter2 = iter(blend_dataset_reload)
    data2 = next(data_iter2)
    for _ in range(iter_num):
    #   print("old", blend_dataset.dataset_states)
    #   print(blend_dataset_reload.dataset_states)
      data1 = next(data_iter1)
      data2 = next(data_iter2)
    #   print("old", blend_dataset.dataset_states)
    #   print(blend_dataset_reload.dataset_states)
      assert deep_compare(data1, data2)


if __name__ == "__main__":
  unittest.main()