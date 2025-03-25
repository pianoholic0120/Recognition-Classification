# Environment Setup

Create virtual environment via `conda`.

```bash
conda create -n yourvenv python==3.8.20
conda activate yourvenv
conda install -c conda-forge cyvlfeat
pip install -r requirements_py38.txt
```

## Dataset Fetch 

    gdown --fuzzy https://drive.google.com/file/d/1drd5FRa5CnUk4ex9Kha_Yae1cNtfU4uB/view\?usp\=drive_link

## File Structure
    |-> cv2025_hw2.pdf
    |-> report.pdf
    |-> requirements_py38.txt
    |-> p1
        |-> bag_of_sift.png
        |-> p1.py
        |-> p1_run.sh
        |-> test_image_feats.pkl
        |-> tiny_image.png
        |-> train_image_feats.pkl
        |-> utils.py
        |-> vocab.pkl
    |-> p2
        |-> config.py
        |-> dataset.py
        |-> download.sh
        |-> model.py
        |-> p2_eval.py
        |-> p2_inference.py
        |-> p2_run_test.sh
        |-> p2_run_train.sh
        |-> p2_train.py
        |-> save_architecture.py
        |-> utils.py
        |-> checkpoint
           |-> mynet_best.pth
           |-> resnet18_best.pth 
        |-> experiment
            |-> mynet_2025_03_24_22_17_52_default
                |-> log
                    |-> config_log.txt
                    |-> result_log.txt
                    |-> train_acc.png
                    |-> train_loss.png
                    |-> val_acc.png
                    |-> val_loss.png
            |-> resnet18_2025_03_25_20_32_50_default
                |-> log
                    |-> config_log.txt
                    |-> result_log.txt
                    |-> train_acc.png
                    |-> train_loss.png
                    |-> val_acc.png
                    |-> val_loss.png
        |-> logs
            |-> mynet_arch.txt
            |-> resnet18_arch.txt
        |-> output 
            |-> pred_mynet.csv
            |-> pred.csv

## Outline
> Part 1: 
- Bag-of-Words Scene Recognition

> Part 2: 
- CNN Image Classification
## Environments

- Python == 3.8.20
- torch==2.2.1
- torchvision==0.17.1
- torchaudio==2.2.1
- matplotlib==3.7.5
- numpy==1.24.0
- Pillow==10.4.0
- scikit-learn==1.3.2
- scipy==1.8.1
- tqdm==4.67.1
- gdown==5.2.0
