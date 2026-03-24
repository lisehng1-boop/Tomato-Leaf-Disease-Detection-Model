from ultralytics import YOLO
import tomato leaf
if __name__ == "__main__":
    model = YOLO('Tl-yolov13')
    model.train(data='',
                epochs=100,
                patience=15,
                lr0=0.01,
                imgsz=640,
                batch=16,
                device=0,
                save=True,
                val=True,
                augment=True,
                mixup=0.5,
                mosaic=1.0
                )
