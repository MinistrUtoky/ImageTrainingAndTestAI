import torch
import cv2
import torchvision.transforms as transforms
import argparse
import torch.nn as nn
import torch.optim as optim
import os
from model import build_model



if __name__ == '__main__':
    paths = ["Dataset/angry/", "Dataset/pleased/", "Dataset/shock/"]
    for path in paths:
        dirs = os.listdir(path)
        c = 0
        count = 0
        for item in dirs:
            device = 'cpu'
            labels = ['angry', 'pleased', 'shock']
            model = build_model(weights=None, fine_tune=False, num_classes=3).to(device)
            checkpoint = torch.load('Outputs/model.pth', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            f, e = os.path.splitext(path + item);
            image = cv2.imread(f + ".png")
            gt_class = (f + ".png").split('/')[-1].split('.')[0]
            orig_image = image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform(image)
            image = torch.unsqueeze(image, 0)
            with torch.no_grad():
                outputs = model(image.to(device))
            output_label = torch.topk(outputs, 1)
            pred_class = labels[int(output_label.indices)]
            cv2.putText(orig_image,
                f"GT: {gt_class}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA
            )
            cv2.putText(orig_image,
                f"Pred: {pred_class}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA
            )
            if (pred_class == path.split("/")[1]):
                count += 1
            print(f"GT: {gt_class}, pred: {pred_class}")
            c+=1
        print(count / c)

