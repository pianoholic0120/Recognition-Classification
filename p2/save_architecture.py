import torch
import torch.nn as nn
import os
from model import MyNet, ResNet18

def print_and_save_model_info(model, model_name, save_path=None):
    architecture_str = str(model)

    param_count = sum(param.numel() for param in model.parameters())

    print(f"===== {model_name} =====")
    print(architecture_str)
    print(f"Total number of parameters: {param_count}")
    print("========================================\n")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(f"===== {model_name} =====\n")
            f.write(architecture_str + '\n')
            f.write(f"Total number of parameters: {param_count}\n")
            f.write("========================================\n\n")


if __name__ == '__main__':
    mynet = MyNet()
    resnet = ResNet18()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mynet.to(device)
    resnet.to(device)

    print_and_save_model_info(mynet, "MyNet", save_path='./logs/mynet_arch.txt')

    print_and_save_model_info(resnet, "ResNet18", save_path='./logs/resnet18_arch.txt')
