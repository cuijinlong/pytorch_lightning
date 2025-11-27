第一步: spectogramer.py -> 转成光谱图 /opt/datasets/spoken/optional/wav 转 /opt/datasets/spoken/optional/spectrograms
第二步：train-test-split.py -> 首先，创建 /opt/datasets/spoken/optional/training-spectrograms 和 /opt/datasets/spoken/optional/testing-spectrograms，然后，将文件名以0-4开头的，作为测试集，其他作为训练集
