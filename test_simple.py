# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets
from networks.configuration import get_config

import networks
from layer import disp_to_depth
from utils import download_model_if_doesnt_exist
from utils import readlines
import matplotlib.pyplot as plt


splits_dir = os.path.join(os.path.dirname(__file__), "splits")
MIN_DEPTH = 1e-3
MAX_DEPTH = 80

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--output_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "RA-Depth"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="png")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    parser.add_argument('--attn-dropout', default=0.0, type=float, metavar='M', help='...')
    parser.add_argument('--ffn-dropout', default=0.0, type=float, metavar='M', help='...')
    parser.add_argument('--dropout', default=0.1, type=float, metavar='M', help='...')
    parser.add_argument('--conv-kernel-size', default=3, type=int, metavar='N', help='...')
    parser.add_argument('--width-multiplier', default=1.0, type=float, metavar='M', help='...')
    parser.add_argument('--backbone-mode', type=str,
                             choices=['small', 'x_small', 'xx_small', 'v0.5', 'v0.75', 'v1.0', 'v1.25'],
                             default='small',
                             help='...')
    parser.add_argument('--backbone', type=str, choices=['mobilevit', 'mobilevitv2'], default='mobilevit',
                             help='...')

    parser.add_argument('--transformer-norm-layer', type=str, default="layer_norm",
                             help="Normalization layer in transformer")
    # parser.add_argument('--no-fuse-local-global-features', action="store_true", help="Do not combine local and global features in MIT block")
    parser.add_argument('--head-dim', type=int, default=None, help="Head dimension in transformer")
    parser.add_argument('--number-heads', type=int, default=4, help="No. of heads in transformer")
    parser.add_argument('--activation-name', type=str, default="prelu", help="...")

    parser.add_argument('--conv-init', type=str, default='kaiming_normal', help='Init type for conv layers')
    parser.add_argument('--conv-init-std-dev', type=float, default=None, help='Std deviation for conv layers')

    parser.add_argument('--linear-init', type=str, default='xavier_uniform', help='Init type for linear layers')
    parser.add_argument('--linear-init-std-dev', type=float, default=0.01, help='Std deviation for Linear layers')

    parser.add_argument('--normalization-name', type=str, default='batch_norm', help='...')
    parser.add_argument('--normalization-momentum', type=float, default=0.1, help='...')



    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    config = get_config(args)

    num_ch_enc = [64, 64, 128, 160, 320]

    encoder = networks.build_model(config)
    #encoder = networks.hrnet18(False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.FusionDecoder(num_ch_enc)



    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict, strict=False)
    depth_decoder.to(device)
    depth_decoder.eval()


    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.output_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            # input_image = input_image.resize((640, 192), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)


            # # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            # name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            # scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
            # np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma') #magma
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_im))

    print('-> Done!')

if __name__ == '__main__':
    args = parse_args()
    test_simple(args)


#CUDA_VISIBLE_DEVICES=0 python test_simple.py --image_path /test/monodepth2-master/assets/test.png --model_name TinyDepth
