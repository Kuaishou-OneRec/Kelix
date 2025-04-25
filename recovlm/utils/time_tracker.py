import time
import os

class TimeTracker:
    def __init__(self, n=1, time_types=["absolute"]):
        """
        初始化 TimeTracker 类。

        :param n: 统计最近 n 次调用的时间间隔平均值
        :param time_types: 时间类型列表，可选值为 "absolute" 或 "cpu"
        """
        self.n = n
        self.time_types = time_types
        self.last_times = {
            "absolute": time.perf_counter(),
            "cpu": os.times().user
        }
        self.interval_records = {}

    def tick(self, name):
        """
        记录指定名称的所有指定时间类型的时间间隔。

        :param name: 时间间隔记录的名称
        """
        for time_type in self.time_types:
            if time_type == "absolute":
                current_time = time.perf_counter()
                last_time = self.last_times["absolute"]
                self.last_times["absolute"] = current_time
            elif time_type == "cpu":
                current_time = os.times().user
                last_time = self.last_times["cpu"]
                self.last_times["cpu"] = current_time
            else:
                raise ValueError("Invalid time_type. Allowed values are 'absolute' or 'cpu'.")

            interval = current_time - last_time

            key = f"{time_type}@{name}"
            if key not in self.interval_records:
                self.interval_records[key] = []

            self.interval_records[key].append(interval)
            if len(self.interval_records[key]) > self.n:
                self.interval_records[key].pop(0)

    def stat(self):
        """
        返回最近 n 次调用的所有时间间隔的平均值。

        :return: 包含每个名称及其平均时间间隔的字典
        """
        result = {}
        for key, intervals in self.interval_records.items():
            if intervals:
                result[key] = sum(intervals) / len(intervals)
        return result
    


def main():
    # 创建一个 TimeTracker 实例，统计最近 3 次调用的时间间隔平均值，记录两种时间类型
    tracker = TimeTracker(n=3, time_types=["absolute", "cpu"])

    # 模拟多次调用 tick 方法
    tracker.tick("event1")
    time.sleep(1)
    tracker.tick("event2")
    time.sleep(2)
    tracker.tick("event1")
    time.sleep(1)
    tracker.tick("event2")

    # 获取统计结果
    statistics = tracker.stat()
    print(statistics)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
        