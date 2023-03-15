import os

import numpy as np
import cv2


img_num = '00231'
image = img_num + '.png'
label = img_num + '.txt'
vi_path = r'./m3fd/vi/'
ir_path = r'./m3fd/ir'
label_path = r'./m3fd/labels/'
vi_path = os.path.join(vi_path, image)
ir_path = os.path.join(ir_path, image)
label_path = os.path.join(label_path, label)


# 坐标转换，原始存储的是YOLOv5格式
# Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


# 读取labels
with open(label_path, 'r') as f:
    # lines = f.readlines()
    # print(lines)
    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
    print(lb)

# 读取图像文件
img = cv2.imread(str(vi_path))
img_ir = cv2.imread(str(ir_path))
h, w = img.shape[:2]
lb[:, 1:] = xywhn2xyxy(lb[:, 1:], w, h, 0, 0)  # 反归一化
print(lb)

classes = ['People', 'Car', 'Bus', 'Lamp', 'Motorcycle', 'Truck']
# 绘图
for _, x in enumerate(lb):
    class_label = classes[int(x[0])]  # class
    print(class_label, ' | ', x[1:].astype(np.int32))
    cv2.rectangle(img, (int(x[1]), int(x[2])), (int(x[3]), int(x[4])), (0, 255, 0))
    cv2.putText(img, str(class_label), (int(x[1]), int(x[2] - 2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,
                color=(0, 0, 255), thickness=1)
    cv2.rectangle(img_ir, (int(x[1]), int(x[2])), (int(x[3]), int(x[4])), (0, 255, 0))
    cv2.putText(img_ir, str(class_label), (int(x[1]), int(x[2] - 2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,
                color=(0, 0, 255), thickness=1)
cv2.imshow('vi', img)
cv2.imshow('ir', img_ir)
cv2.waitKey(0)  # 按键结束
cv2.destroyAllWindows()
