# Top 1 Zalo AI challenge 2021 5K-Compliance 

## Prepare data: Dữ liệu của ban tổ chức cung cấp nằm trong folder dataset/train. Kết quả sau khi chia chính xác như các file txt
- Clone yolov5 và mô hình yolov5x6 và thay thế file detect.py
```bash
git clone https://github.com/ultralytics/yolov5
scp detect.py yolov5
cd yolov5
wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5x6.pt
python detect.py --weights yolov5x6.pt --source /dataset/train/images --img 1536 --conf-thres 0.1 --iou-thres 0.7 --nosave --classes 0 --augment
cd ..
```
- Chia dữ liệu 5-fold
```bash
python3 prepare_mask_dataset.py
python3 prepare_distance_dataset.py
```


## train mask model
```bash
bash script_train_mask/runtrain1.sh
bash script_train_mask/runtrain2.sh
bash script_train_mask/runtrain3.sh
bash script_train_mask/runtrain4.sh
bash script_train_mask/runtrain5.sh
```
## train distance model
```bash
bash script_train_distance/runtrain1.sh
bash script_train_distance/runtrain2.sh
bash script_train_distance/runtrain3.sh
bash script_train_distance/runtrain4.sh
bash script_train_distance/runtrain5.sh
```


## inference
```bash
bash predict.sh
```
