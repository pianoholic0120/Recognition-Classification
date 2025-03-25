### Manually put your best model under folder checkpoint/ and rename as 'resnet18_best.pth' or 'mynet_best.pth'
python3 p2_inference.py --test_datadir ../hw2_data/p2_data/val --model_type resnet18 --output_path ./output/pred.csv
# python3 p2_inference.py --test_datadir ../hw2_data/p2_data/val --model_type mynet --output_path ./output/pred_mynet.csv

python3 p2_eval.py --csv_path ./output/pred.csv --annos_path ../hw2_data/p2_data/val/annotations.json