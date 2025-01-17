import sys
import json
from recovlm.utils.common import shell_hdfs_ls

if __name__ == "__main__":
    data_folder = sys.argv[1]
    json_file = sys.argv[2]

    fn_list = shell_hdfs_ls(data_folder)
    fn_list = [fn for fn in fn_list if fn.endswith(".parquet")]

    with open(json_file, "w+") as fp:
        fp.write(json.dumps(fn_list))