第一步：
1、下载 npz 并转换PathMNIST数据集
    |
    -- 下载目录：/opt/datasets/pathmnist
2、模拟Excel数据
    |
    -- /opt/datasets/pathmnist
第二步：
1、原始数据集的目录结构
/opt/datasets/pifubing
   |
    -- optional_image    # 样本数据
         |  # 不同分类图片文件夹
         -- /cate1
         -- /cate2
         -- /cate3
         -- ...
         -- /caten
         |
         --data.xlsx  # 可选（可有可无）

2、分割数据集的目录结构(无xlsx)
/opt/datasets/pifubing
   |
    --output_dataset_basic
         |  # 不同分类图片文件夹
         -- /test
             -- /cate1
             -- /cate2
             -- /cate3
             -- ...
             -- /caten
         -- /train
             -- /cate1
             -- /cate2
             -- /cate3
             -- ...
             -- /caten
         -- /val
             -- /cate1
             -- /cate2
             -- /cate3
             -- ...
             -- /caten
         --test_metadata.csv
         --train_metadata.csv
         --val_metadata.csv
         --dataset_info.txt

3、分割数据集的目录结构(有xlsx)
/opt/datasets/pifubing
   |
    --output_dataset_multimodal
         |
         -- /test
             -- /cate1
             -- /cate2
             -- /cate3
             -- ...
             -- /caten
         -- /train
             -- /cate1
             -- /cate2
             -- /cate3
             -- ...
             -- /caten
         -- /val
             -- /cate1
             -- /cate2
             -- /cate3
             -- ...
             -- /caten
         --test_metadata.csv
         --train_metadata.csv
         --val_metadata.csv
         --dataset_info.txt
