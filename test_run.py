import os

os.system('python ./train.py --config configs/hardnet_changwon.yml')
os.system('python ./validate.py --config configs/hardnet_changwon.yml --model_path /workspace/jihun/yswang/Demo/Demo/FCHarDNet/pretrained/hardnet_add_model.pkl')
os.system('python ./visualize.py --config configs/hardnet_changwon.yml --model_path /workspace/jihun/yswang/Demo/Demo/FCHarDNet/pretrained/hardnet_add_model.pkl')
