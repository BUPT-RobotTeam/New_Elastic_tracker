#!/usr/bin/env python
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import rospy
import cv2
import numpy as np
import torch
import os
import sys
import std_msgs
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import Detection2D
from module.detector import Detector
from utils.tool import handle_preds


# 自定义图像转换函数，替代cv_bridge
def imgmsg_to_cv2(img_msg, desired_encoding='bgr8'):
    """将ROS Image消息转换为OpenCV格式"""
    if desired_encoding == 'bgr8':
        dtype = np.dtype("uint8")
        channels = 3
    else:
        rospy.logerr(f"不支持的编码格式: {desired_encoding}")
        return None

    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    try:
        image_cv = np.ndarray(
            shape=(img_msg.height, img_msg.width, channels),
            dtype=dtype,
            buffer=img_msg.data
        )
    except Exception as e:
        rospy.logerr(f"图像转换失败: {str(e)}")
        return None

    if img_msg.is_bigendian != (sys.byteorder == 'little'):
        image_cv = image_cv.byteswap().newbyteorder()

    return image_cv

def cv2_to_imgmsg(cv_image, encoding='bgr8'):
    """将OpenCV图像转换为ROS Image消息"""
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = encoding
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tobytes()
    img_msg.step = len(img_msg.data) // img_msg.height
    return img_msg

target_classes = ["person"]


path_curr = os.path.dirname(__file__)
img_topic = "/iris_0/realsense/depth_camera/color/image_raw"
device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备
weight = "weights/weight.pth"
class_names = "config/coco.names"
thresh = 0.65

class yolo_detector:
    def __init__(self):
        print("[onboardDetector]: yolo detector init...")
        print(f"[onboardDetector]: 使用设备: {device}")

        self.img_received = False
        self.img_detected = False

        # 验证权重文件是否存在
        weight_path = os.path.join(path_curr, weight)
        if not os.path.exists(weight_path):
            rospy.logfatal(f"权重文件不存在: {weight_path}")
            rospy.signal_shutdown("权重文件缺失")
            return

        # 初始化并加载完整模型
        try:
            print(f"[onboardDetector]: 加载模型权重: {weight_path}")
            
            # 加载完整的模型 checkpoint
            checkpoint = torch.load(weight_path, map_location=device)
            
            # 检查checkpoint是否包含模型结构
            if 'model' in checkpoint:
                # 如果checkpoint包含模型对象，从模型对象中获取权重
                self.model = Detector(80, True).to(device)
                model_weights = checkpoint['model'].state_dict()
                self.model.load_state_dict(model_weights, strict=False)
            else:
                # 否则直接加载权重字典
                self.model = Detector(80, True).to(device)
                self.model.load_state_dict(checkpoint, strict=False)
                
            self.model.eval()
            print("[onboardDetector]: 模型加载成功")
        except Exception as e:
            rospy.logfatal(f"模型加载失败: {str(e)}")
            rospy.signal_shutdown("模型加载失败")
            return

        # 订阅者
        self.img_sub = rospy.Subscriber(img_topic, Image, self.image_callback)

        # 发布者
        self.img_pub = rospy.Publisher("yolo_detector/detected_image", Image, queue_size=10)
        self.bbox_pub = rospy.Publisher("yolo_detector/detected_bounding_boxes", Detection2DArray, queue_size=10)
        self.time_pub = rospy.Publisher("yolo_detector/yolo_time", std_msgs.msg.Float64, queue_size=1)

        # 定时器
        rospy.Timer(rospy.Duration(0.033), self.detect_callback)
        rospy.Timer(rospy.Duration(0.033), self.vis_callback)
        rospy.Timer(rospy.Duration(0.033), self.bbox_callback)
    
    def image_callback(self, msg):
        # 使用自定义转换函数替代CvBridge
        self.img = imgmsg_to_cv2(msg, "bgr8")
        self.img_received = True

    def detect_callback(self, event):
        startTime = rospy.Time.now()
        if self.img_received:
            output = self.inference(self.img)
            self.detected_img, self.detected_bboxes = self.postprocess(self.img.copy(), output)  # 使用副本避免原图被修改
            self.img_detected = True
        endTime = rospy.Time.now()
        self.time_pub.publish((endTime-startTime).to_sec())
        

    def vis_callback(self, event):
        if self.img_detected:
            # 使用自定义转换函数替代CvBridge
            self.img_pub.publish(cv2_to_imgmsg(self.detected_img, "bgr8"))

    def bbox_callback(self, event):
        if self.img_detected:
            bboxes_msg = Detection2DArray()
            bboxes_msg.header.stamp = rospy.Time.now()  # 统一设置时间戳
            for detected_box in self.detected_bboxes:
                if detected_box[4] in target_classes:
                    bbox_msg = Detection2D()
                    bbox_msg.bbox.center.x = int((detected_box[0] + detected_box[2]) / 2)  # 计算中心x坐标
                    bbox_msg.bbox.center.y = int((detected_box[1] + detected_box[3]) / 2)  # 计算中心y坐标
                    bbox_msg.bbox.size_x = abs(detected_box[2] - detected_box[0]) 
                    bbox_msg.bbox.size_y = abs(detected_box[3] - detected_box[1])

                    bboxes_msg.detections.append(bbox_msg)
            self.bbox_pub.publish(bboxes_msg)

    def inference(self, ori_img):
        # 图像预处理
        res_img = cv2.resize(ori_img, (352, 352), interpolation = cv2.INTER_LINEAR) 
        img = res_img.reshape(1, 352, 352, 3)
        img = torch.from_numpy(img.transpose(0, 3, 1, 2))
        img = img.to(device).float() / 255.0    

        # 推理
        with torch.no_grad():  # 禁用梯度计算，提高速度并减少内存使用
            preds = self.model(img)
            output = handle_preds(preds, device, thresh)
        return output

    def postprocess(self, ori_img, output):
        LABEL_NAMES = []
        class_names_path = os.path.join(path_curr, class_names)
        if not os.path.exists(class_names_path):
            rospy.logwarn(f"类别名称文件不存在: {class_names_path}")
            return ori_img, []
            
        with open(class_names_path, 'r') as f:
            for line in f.readlines():
                LABEL_NAMES.append(line.strip())
        
        H, W, _ = ori_img.shape
        
        detected_boxes = []
        if output and len(output) > 0:
            for box in output[0]:
                box = box.tolist()
                
                obj_score = box[4]
                category_idx = int(box[5])
                # 检查类别索引是否有效
                if category_idx < 0 or category_idx >= len(LABEL_NAMES):
                    category = f"未知类别({category_idx})"
                else:
                    category = LABEL_NAMES[category_idx]
                    
                # 计算原始图像上的边界框坐标
                x1, y1 = int(box[0] * W), int(box[1] * H)
                x2, y2 = int(box[2] * W), int(box[3] * H)
                
                # 确保坐标在图像范围内
                x1 = max(0, min(x1, W-1))
                y1 = max(0, min(y1, H-1))
                x2 = max(0, min(x2, W-1))
                y2 = max(0, min(y2, H-1))
                
                detected_box = [x1, y1, x2, y2, category]
                detected_boxes.append(detected_box)

                # 绘制边界框和标签
                cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(ori_img, f'{category} {obj_score:.2f}', 
                           (x1, max(10, y1 - 5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        return ori_img, detected_boxes

def main():
    rospy.init_node('yolo_detector_node', anonymous=True)
    try:
        detector = yolo_detector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
