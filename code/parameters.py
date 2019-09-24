import argparse
import datetime
import os

def get_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataroot', type=str, default='../input/',
                        help='Path to root of image data (saved in dirs of classes)')

    parser.add_argument('--save_model_dir', type=str, default='../sagan_models/')

    parser.add_argument('--sample_images_path', type=str, default='../output_images/')

    parser.add_argument('--submission_dir', type=str, default='../submission/')


    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epoches', type=int, default=170)
    parser.add_argument('--d_lr', type=float, default=0.0004)
    parser.add_argument('--g_lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--real_label_value', type=float, default=0.8)
    parser.add_argument('--fake_label_value', type=float, default=0)

    parser.add_argument('--adv_loss', type=str, default='dcgan')
    parser.add_argument('--z_dim', type=int, default=180)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)

    parser.add_argument('--imsize', type=int, default=64)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_of_classes', type=int, default=120)
    parser.add_argument('--num_sample_images', type=int, default=10000)

    args = parser.parse_args()

    return args