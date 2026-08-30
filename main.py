from ultralytics import YOLO
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE, 'datasets', 'data.yaml')

def yolo():
    model = YOLO("yolo26m.pt")    # YOLO26m 目标检测模型

    print("测验开始")

    result = model.train(
        data=DATA_YAML,
        epochs=10000,   # 训练轮次
        imgsz=940,      # 输入图片大小
        patience=100,   # 等待响应100
        batch=32,       # 一次训练32张图片
        save=True,      # 保存训练权重
        save_period=100,  # 每100轮保存一次
        plots=True,
        val=True,
        workers=0,
        device=0,       # 使用第0块GPU
        amp=True,
        name="AI COMP",
    )

    print("测验完成")
    print(result)



if __name__ == "__main__":
    yolo()
