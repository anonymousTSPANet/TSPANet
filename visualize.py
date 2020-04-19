import yaml
import torch
import argparse
import timeit
import time
import os
import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt

from torch.utils import data
from torchstat import stat
from pytorch_bn_fusion.bn_fusion import fuse_bn_recursively

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict

import cv2
from PIL import Image

torch.backends.cudnn.benchmark = True


def reset_batchnorm(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.reset_running_stats()
        m.momentum = None
#
#
# colors = [  # [  0,   0,   0],
#     [128, 64, 128],
#     [244, 35, 232],
#     [70, 70, 70],
#     [102, 102, 156],
#     [190, 153, 153],
#     [153, 153, 153],
#     [250, 170, 30],
#     [220, 220, 0],
#     [107, 142, 35],
#     [152, 251, 152],
#     [0, 130, 180],
#     [220, 20, 60],
#     [255, 0, 0],
#     [0, 0, 142],
#     [0, 0, 70],
#     [0, 60, 100],
#     [0, 80, 100],
#     [0, 0, 230],
#     [119, 11, 32],
# ]
#
# label_colours = dict(zip(range(19), colors))

# colors = [ (170, 170, 170), # concrete normal 0
#                (213, 213, 213), # concrete icy 1
#                (100, 100, 100), # concrete wet 2
#                (0, 255, 0), # grass normal 3
#                (100, 255, 100), # grass icy 4
#                (47, 157, 39), # grass wet 5
#                (128, 64, 0), # gravel normal 6
#                (137, 137, 85), # gravel icy 7
#                (202, 202, 126), # gravel wet 8
#                (200, 150, 80), # dirt normal 9
#                (230, 180, 110), # dirt icy 10
#                (170, 120, 50), # dirt wet 11
#                (255, 255, 0), # puddle normal 12
#                (255, 255, 150), # puddle icy 13
#                (220, 220, 120), # puddle wet 14
#                (80, 200, 222), # block normal 15
#                (110, 255, 255), # 16
#                (40, 160, 190), # 17
#                (250, 100, 165), # asphalt normal 18
#                (255, 180, 220), # 19
#                (220, 60, 140), # 20
#                (200, 86, 20), # openfields normal # 21
#                (73, 73, 45), # 22
#                (80, 40, 15), # 23
#                (217, 229, 255),  # snow # 24
#                (0, 120, 255), # sky 25
#                (0, 60, 0), # forest 26
#                (102, 102, 51), # mountain 27
#                (128, 64, 128), # artificial 28
#                (0, 0, 0), # void 29
#                (255, 94, 0), # soldier 30
#                (255, 187, 0), # tank 31
#                (167, 72, 255), # slope 32
#                (133, 196, 193), # person 33
#                (140, 140, 140)] # trash 34

colors = [(170, 170, 170),  # concrete normal 0
          # (213, 213, 213),  # concrete icy 1
          # (100, 100, 100),  # concrete wet 2
          (0, 255, 0),  # grass normal 3
          # (100, 255, 100),  # grass icy 4
          # (47, 157, 39),  # grass wet 5
          (128, 64, 0),  # gravel normal 6
          # (137, 137, 85),  # gravel icy 7
          # (202, 202, 126),  # gravel wet 8
          #(200, 150, 80),  # dirt normal 9
          # (230, 180, 110),  # dirt icy 10
          # (170, 120, 50),  # dirt wet 11
          #(255, 255, 0),  # puddle normal 12
          # (255, 255, 150),  # puddle icy 13
          # (220, 220, 120),  # puddle wet 14
          #(80, 200, 222),  # block normal 15
          # (110, 255, 255),  # 16
          # (40, 160, 190),  # 17
          #(250, 100, 165),  # asphalt normal 18
          # (255, 180, 220),  # 19
          # (220, 60, 140),  # 20
          #(200, 86, 20),  # openfields normal # 21
          # (73, 73, 45),  # 22
          # (80, 40, 15),  # 23
          #(217, 229, 255),  # snow # 24
          (0, 120, 255),  # sky 25
          (0, 60, 0),  # forest 26
          #(102, 102, 51),  # mountain 27
          (128, 64, 128),  # artificial 28
          #(0, 0, 0),  # void 29
          #(255, 94, 0),  # soldier 30
          #(255, 187, 0),  # tank 31
          #(167, 72, 255),  # slope 32
          #(133, 196, 193),  # person 33
          #(140, 140, 140)]  # trash 34
          ]


label_colours = dict(zip(range(len(colors)), colors))
print(label_colours)

def decode_segmap(temp):
    print('temp shape : {}'.format(temp.shape))
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()

    #valid_classes = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    #map_pred_to_val = dict(zip(range(len(valid_classes)), valid_classes))

    for l in range(0, 6):
        #l = map_pred_to_val[l]
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    # rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    # rgb[:, :, 0] = r / 255.0
    # rgb[:, :, 1] = g / 255.0
    # rgb[:, :, 2] = b / 255.0

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    return rgb


def get_transformed_img(img_path,img_size):
    img = Image.open(img_path)
    img = np.array(img, dtype=np.uint8)

    #img_size = (1024, 2048)
    #img_size = (1080,1920)
    #img_size=(360,360)
    #img_size = (480, 640)
    img = np.array(Image.fromarray(img).resize(
        (img_size[1], img_size[0])))  # uint8 with RGB mode
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)

    value_scale = 255
    mean = [0.406, 0.456, 0.485]
    mean = [item * value_scale for item in mean]
    std = [0.225, 0.224, 0.229]
    std = [item * value_scale for item in std]

    img_norm = True
    if img_norm:
        img = (img - mean) / std

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)

    # expand batch (jh)
    img = np.expand_dims(img, axis=0)

    img = torch.from_numpy(img).float().to(torch.device('cuda'))
    return img


def validate_single(cfg, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    # loader = data_loader(
    #     data_path,
    #     split=cfg["data"]["val_split"],
    #     is_transform=True,
    #     img_size=(1024, 2048),
    # )

    n_classes = 6

    # valloader = data.DataLoader(loader, batch_size=1, num_workers=1)
    running_metrics = runningScore(n_classes)

    # Setup Model

    model = get_model(cfg["model"], n_classes).to(device)

    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)

    if args.bn_fusion:
        model = fuse_bn_recursively(model)
        print(model)

    if args.update_bn:
        print("Reset BatchNorm and recalculate mean/var")
        model.apply(reset_batchnorm)
        model.train()
    else:
        model.eval()
    model.to(device)
    total_time = 0

    total_params = sum(p.numel() for p in model.parameters())
    print('Parameters: ', total_params)

    # stat(model, (3, 1024, 2048))
    torch.backends.cudnn.benchmark = True

    #target_dir = '/workspace/raid/driving/Data/CITYSCAPES/leftImg8bit/val/frankfurt'
    #target_dir = '/workspace/raid/driving/Data/segmentation_changwon/640_480/img/'
    #target_dir = '/workspace/raid/driving/Data/segmentation_changwon/'+cfg['data']['val_split']+'/img/'
    target_dir = '/workspace/raid/driving/Data/segmentation_changwon/640_480/img'
    target_dir = os.path.join(cfg['data']['path'],cfg['data']['val_split'],'img')
    img_size = (cfg["data"]["img_rows"], cfg["data"]["img_cols"])
    for target_img_path in os.listdir(target_dir):
        org_image = cv2.imread(os.path.join(target_dir, target_img_path))
        org_image = np.array(Image.fromarray(org_image).resize(
            (img_size[1], img_size[0])))  # uint8 with RGB mode

        # image = np.expand_dims(org_image, axis=0)
        # image = torch.from_numpy(image).float().to(device)
        # image = image.permute(0, 3, 1, 2)

        image = get_transformed_img(os.path.join(target_dir, target_img_path),img_size)


        with torch.no_grad():
            outputs = model(image)
            outputs = outputs.permute(0, 2, 3, 1)
            # pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
            pred = np.squeeze(outputs.data.max(3)[1].cpu().numpy(), axis=0)

            decoded = decode_segmap(pred)
            # img_input = np.squeeze(images.cpu().numpy(), axis=0)
            # img_input = img_input.transpose(1, 2, 0)
            # blend = img_input * 0.2 + decoded * 0.8
            print(org_image.shape, decoded.shape)
            blend = cv2.addWeighted(org_image, 0.8, decoded, 0.2, 0)

            decoded = decoded[:, :, ::-1]  # RGB -> BGR
            '''
            fig = plt.figure(figsize=(10,10))
            rows = 1; cols = 3; img=[decoded,org_image,blend]
            xlabels = [" ", "predicted","org", "blend"]
            for i in range(1,4):
                ax = fig.add_subplot(rows,cols,i)
                ax.imshow(cv2.cvtColor(img[i-1],cv2.COLOR_BGR2RGB))
                ax.set_xlabel(xlabels[i])
                ax.set_xticks([]), ax.set_yticks([])
            plt.show()
            '''

            cv2.imshow('predicted', decoded)
            cv2.imshow('org', org_image)
            cv2.imshow('blend', blend)
            #
            cv2.waitKey(0)

            # cv2.imwrite(os.path.join('output_200113', target_img_path[:-4]+"_pred.png"), decoded)
            # cv2.imwrite(os.path.join('output_200113', target_img_path[:-4] + "_org.png"), org_image)
            # cv2.imwrite(os.path.join('output_200113', target_img_path[:-4] + "_blend.png"), blend)


def validate(cfg, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]
    img_size = (cfg["data"]["img_rows"],cfg["data"]["img_cols"])
    loader = data_loader(
        data_path,
        split=cfg["data"]["val_split"],
        is_transform=True,
        img_size=img_size,
    )

    n_classes = loader.n_classes
    valloader = data.DataLoader(loader, batch_size=1, num_workers=1)
    running_metrics = runningScore(n_classes)

    # Setup Model

    model = get_model(cfg["model"], n_classes).to(device)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)

    if args.bn_fusion:
        model = fuse_bn_recursively(model)
        print(model)

    if args.update_bn:
        print("Reset BatchNorm and recalculate mean/var")
        model.apply(reset_batchnorm)
        model.train()
    else:
        model.eval()
    model.to(device)
    total_time = 0

    total_params = sum(p.numel() for p in model.parameters())
    print('Parameters: ', total_params)

    # stat(model, (3, 1024, 2048))
    torch.backends.cudnn.benchmark = True

    for i, (images, labels, fname) in enumerate(valloader).sort(reverse=True):
        start_time = timeit.default_timer()

        images = images.to(device)

        if i == 0:
            with torch.no_grad():
                outputs = model(images)

        if args.eval_flip:
            outputs = model(images)

            # Flip images in numpy (not support in tensor)
            outputs = outputs.data.cpu().numpy()
            flipped_images = np.copy(images.data.cpu().numpy()[:, :, :, ::-1])
            flipped_images = torch.from_numpy(flipped_images).float().to(device)
            outputs_flipped = model(flipped_images)
            outputs_flipped = outputs_flipped.data.cpu().numpy()
            outputs = (outputs + outputs_flipped[:, :, :, ::-1]) / 2.0

            pred = np.argmax(outputs, axis=1)
        else:
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                outputs = model(images)

            torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - start_time

            if args.save_image:
                pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
                save_rgb = True

                decoded = loader.decode_segmap_id(pred)
                dir = "./out_predID/"
                if not os.path.exists(dir):
                    os.mkdir(dir)
                misc.imsave(dir + fname[0], decoded)

                if save_rgb:
                    decoded = loader.decode_segmap(pred)
                    img_input = np.squeeze(images.cpu().numpy(), axis=0)
                    img_input = img_input.transpose(1, 2, 0)
                    blend = img_input * 0.2 + decoded * 0.8
                    fname_new = fname[0]
                    fname_new = fname_new[:-4]
                    fname_new += '.jpg'
                    dir = "./out_rgb/"
                    if not os.path.exists(dir):
                        os.mkdir(dir)
                    misc.imsave(dir + fname_new, blend)

            pred = outputs.data.max(1)[1].cpu().numpy()

        gt = labels.numpy()
        #s = np.sum(gt == pred) / (1024 * 2048)
        s = np.sum(gt == pred) / (img_size[0]*img_size[1])

        if args.measure_time:
            total_time += elapsed_time
            print(
                "Inference time \
                  (iter {0:5d}): {1:4f}, {2:3.5f} fps".format(
                    i + 1, s, 1 / elapsed_time
                )
            )

        running_metrics.update(gt, pred)

    score, class_iou = running_metrics.get_scores()
    print("Total Frame Rate = %.2f fps" % (500 / total_time))

    if args.update_bn:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        state2 = {"model_state": model.state_dict()}
        torch.save(state2, 'hardnet_cityscapes_mod.pth')

    for k, v in score.items():
        print(k, v)

    for i in range(n_classes):
        print(i, class_iou[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/test/hardnet_nl_changwon_exp7.yml",
        #default="configs/test/hardnet.yml",
        help="Config file to be used",
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        # default="hardnet_cityscapes_best_model.pkl",
        #default="/workspace/raid/yswang/driving_segmentation/FCHarDNet/runs/hardnet_changwon/cur/hardnet_add_best_model_360.pkl",
        #default="/workspace/raid/yswang/driving_segmentation/FCHarDNet/runs/hardnet_nl_changwon_test/cur/hardnet_nl_add_checkpoint_hardnet_nl_changwon_1.pkl",
        default="/workspace/raid/yswang/driving_segmentation/FCHarDNet/runs/hardnet_nl_changwon_test/cur/hardnet_nl_add_best_model_hardnet_nl_allset_1.pkl",
        #default="./hardnet70_cityscapes_model.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--eval_flip",
        dest="eval_flip",
        action="store_true",
        help="Enable evaluation with flipped image |\
                              False by default",
    )
    parser.add_argument(
        "--no-eval_flip",
        dest="eval_flip",
        action="store_false",
        help="Disable evaluation with flipped image",
    )
    parser.set_defaults(eval_flip=False)

    parser.add_argument(
        "--measure_time",
        dest="measure_time",
        action="store_true",
        help="Enable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.add_argument(
        "--no-measure_time",
        dest="measure_time",
        action="store_false",
        help="Disable evaluation with time (fps) measurement",
    )
    parser.set_defaults(measure_time=True)

    parser.add_argument(
        "--save_image",
        dest="save_image",
        action="store_true",
        help="Enable saving inference result image into out_img/ |\
                              False by default",
    )
    parser.set_defaults(save_image=False)

    parser.add_argument(
        "--update_bn",
        dest="update_bn",
        action="store_true",
        help="Reset and update BatchNorm running mean/var with entire dataset |\
              False by default",
    )
    parser.set_defaults(update_bn=False)

    parser.add_argument(
        "--no-bn_fusion",
        dest="bn_fusion",
        action="store_false",
        help="Disable performing batch norm fusion with convolutional layers |\
              bn_fusion is enabled by default",
    )
    parser.set_defaults(bn_fusion=True)

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    # validate(cfg, args)
    validate_single(cfg, args)
