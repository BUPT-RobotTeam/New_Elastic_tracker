import torch
 
# 将pt文件转换为pth文件
pt_file_path = '/home/wsnc/Fast-Tracker-main/src/d2p/scripts/yolo_detector/weights/yolov8s.pt'
pth_file_path = '/home/wsnc/Fast-Tracker-main/src/d2p/scripts/yolo_detector/weights/weight1.pth'
model_weights = torch.load(pt_file_path)
torch.save(model_weights, pth_file_path)
