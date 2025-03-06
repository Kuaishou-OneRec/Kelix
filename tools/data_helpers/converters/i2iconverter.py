import os
import random
from typing import Dict, List, Tuple, Set

class i2iconverter:
    def __init__(self, textfile_path: str):
        self.textfile_path = textfile_path

    def read_textfile(self) -> List[Tuple[str, str, float]]:
        with open(self.textfile_path, 'r') as file:
            lines = file.readlines()
        data = []
        for line in lines:
            src_pid, sim_pid, sim_score = line.strip().split('\t')
            sim_score = float(sim_score)
            if '152189812003' <= src_pid <= '157862391293' and '152189812003' <= sim_pid <= '157862391293':
                data.append((src_pid, sim_pid, sim_score))
        # Sort by similarity score in descending order
        data.sort(key=lambda x: x[2], reverse=True)
        unique_src_data = []
        seen_src_pids = set()
        for entry in data:
            src_pid = entry[0]
            if src_pid not in seen_src_pids:
                unique_src_data.append(entry)
                seen_src_pids.add(src_pid)
            if len(unique_src_data) == 1000:
                break
        return unique_src_data

    def sample_neg_pid(self, src_pid: str, sim_pids: List[str]) -> str:
        sim_pids = [pid for pid in sim_pids if pid != src_pid]
        return random.choice(sim_pids) if sim_pids else None

    def process(self) -> Tuple[List[Tuple[str, str, str]], Set[str]]:
        data = self.read_textfile()
        video_pool = set()
        triplets = []
        for src_pid, sim_pid, _ in data:
            video_pool.add(src_pid)
            video_pool.add(sim_pid)
        for src_pid, sim_pid, _ in data:
            neg_pid = self.sample_neg_pid(src_pid, list(video_pool))
            if neg_pid:
                triplets.append((src_pid, sim_pid, neg_pid))

        # Save triplets to a file
        with open('triplets', 'w') as triplets_file:
            for triplet in triplets:
                triplets_file.write('\t'.join(triplet) + '\n')

        # Save video pool to a file
        with open('video_pool', 'w') as video_pool_file:
            for pid in video_pool:
                video_pool_file.write(pid + '\n')

        return triplets, video_pool

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 i2iconverter.py <textfile_path>")
        sys.exit(1)
    textfile_path = sys.argv[1]
    converter = i2iconverter(textfile_path)
    converter.process() 