import os
from pathlib import Path
import shutil

import torch
from torchvision.utils import save_image

import parameters
import self_atten_gan
import utils

def save_images(generator_model, sample_images_path, submission_dir, z_dim, num_of_classes=120, num_images=10000):
    sample_images_dir = Path(sample_images_path)
    sample_images_dir.mkdir(exist_ok=True)

    im_batch_size = 50
    device = torch.device('cuda')

    for i_batch in range(0, num_images, im_batch_size):
        z = utils.truncated_normal((im_batch_size, z_dim), threshold=1)
        gen_z = torch.from_numpy(z).float().to(device)
        dog_labels = torch.squeeze(torch.randint(0, num_of_classes, (im_batch_size,), device=device))
        gen_images = generator_model(gen_z, dog_labels)
        images = gen_images.to('cpu').clone().detach()
        images = images.numpy().transpose(0, 2, 3, 1)
        for i_image in range(gen_images.size(0)):
            save_image(utils.denorm(gen_images[i_image, :, :, :]), sample_images_dir / f'image_{i_batch + i_image:05d}.png')

    submission_dir = Path(submission_dir)
    submission_dir.mkdir(exist_ok=True)
    shutil.make_archive(os.path.join(submission_dir, 'images'), 'zip', sample_images_path)

if __name__ == '__main__':
    args = parameters.get_parameters()

    seed = 1234
    utils.seed_everything(seed)

    # load generator model
    device = torch.device('cuda')
    gen_model = self_atten_gan.Generator(args.z_dim, args.g_conv_dim, args.num_of_classes).to(device)
    gen_model.load_state_dict(torch.load(os.path.join(args.save_model_dir, 'gen_model.pth')))
    save_images(generator_model=gen_model,
                sample_images_path=args.sample_images_path,
                submission_dir=args.submission_dir,
                z_dim=args.z_dim,
                num_of_classes=args.num_of_classes,
                num_images=args.num_sample_images)