import cv2
import torch
import time
import glob
import numpy as np
from ultralytics import YOLO

# Model is trained on 640x640 pics code crops all pics to this size
PicSize = (640, 640)
ObjectPath = "Object/demo.jpg"
modelPath = "N40Ep_Seg.pt"


def MiDasInit():
    # There are bigger models but they are slower
    # model_type = "DPT_Large"
    # model_type = "DPT_Hybrid"
    model_type = "MiDaS_small"

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    # Move model to GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Cuda")
    else:
        device = torch.device("cpu")
        print("Cpu")
    midas.to(device)
    midas.eval()
    # Load transforms to resize and normalize the image
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    return midas, transform, device


def MiDas(ObjectPath, midas, transform, device):
    img = cv2.imread(ObjectPath)
    img = cv2.resize(img, PicSize)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run picture through MiDas if possible gpu else cpu
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    depth_map = cv2.normalize(
        depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    depth_map = (depth_map * 255).astype(np.uint8)

    return img, depth_map


def ObjectDetect(img, depth, model):
    result = model.predict(
        source=img, show=False, conf=0.3, retina_masks=True, boxes=False
    )
    # Make result segemented object visible
    depth_sig = np.stack((depth, depth, depth), axis=2)
    Sigmented = result[0].plot(conf=False, labels=False, boxes=False, img=depth_sig)
    Sigmented = np.stack((depth, depth, Sigmented[:, :, 2]), axis=2)

    # Make object more red but loses Midas depth for red object
    red = np.where(Sigmented[:, :, 2] > Sigmented[:, :, 0])
    Sigmented[red] = [0, 0, 255]
    return Sigmented


def Show(depht, sig, img):
    cv2.imshow("depht", depht)
    cv2.imshow("sig", sig)
    cv2.imshow("img", img)
    while True:
        # Press "q" to quit windows
        if cv2.waitKey(5) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break


def main():
    model = YOLO(modelPath)
    PicPath = glob.glob(ObjectPath)
    midas, transform, device = MiDasInit()

    for Pic in PicPath:
        start = time.time()
        img, depth = MiDas(Pic, midas, transform, device)
        sig = ObjectDetect(img, depth, model)
        end = time.time()
        Show(depth, sig, img)
        print("Time: ", end - start)


if __name__ == "__main__":
    main()
