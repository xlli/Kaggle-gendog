import os
import random
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from tqdm import tqdm

import parameters
import self_atten_gan
import data_processing
import utils

def train_model(args):
    train_dataset = data_processing.get_train_dataset(args.dataroot, args.imsize)
    num_of_classes = len(set(train_dataset.classes))
    train_dataloader = data_processing.get_dataloader(train_dataset, args.batch_size, num_workers=4)

    device = torch.device('cuda')

    G = self_atten_gan.Generator(args.z_dim, args.g_conv_dim, num_of_classes).to(device)
    D = self_atten_gan.Discriminator(args.d_conv_dim, num_of_classes).to(device)

    G_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, G.parameters()), args.g_lr,
                                   [args.beta1, args.beta2])
    D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), args.d_lr,
                                   [args.beta1, args.beta2])

    lr_schedulerG = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(G_optimizer, T_0=args.num_epoches // 10,
                                                                         eta_min=0.00001)
    lr_schedulerD = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(D_optimizer, T_0=args.num_epoches // 10,
                                                                         eta_min=0.00005)

    if args.adv_loss == 'dcgan':
        criterion = nn.BCELoss()

    cudnn.benchmark = True

    G.train()
    D.train()

    label = torch.full((args.batch_size,), 1, device=device)
    ones = torch.full((args.batch_size,), 1, device=device)

    gen_losses = []
    dis_losses = []

    for epoch in range(args.num_epoches):
        #print("\nEpoch: %d" % epoch)

        for i, (real_images, dog_labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            real_images = real_images.to(device)
            dog_labels = torch.tensor(dog_labels, device=device)

            # =================================== TRAIN D ============================== #
            G_optimizer.zero_grad()
            D_optimizer.zero_grad()

            # TRAIN with REAL
            # Get D output for real images & real labels
            d_out_real = D(real_images, dog_labels)

            # Compute D loss with real images & real_labels
            label.fill_(args.real_label_value) + np.random.uniform(-0.1, 0.1)
            d_loss_real = criterion(torch.sigmoid(d_out_real), label)

            # Backward
            d_loss_real.backward()

            # Train with FAKE
            # Create random noise
            z = torch.randn(args.batch_size, args.z_dim, device=device)

            # Generate fake images for same dog labels
            fake_images = G(z, dog_labels)

            # Get D output for fake images & same dog labels
            d_out_fake = D(fake_images.detach(), dog_labels)

            # Compute D loss with fake images & real labels
            label.fill_(args.fake_label_value) + np.random.uniform(0, 0.2)
            d_loss_fake = criterion(torch.sigmoid(d_out_fake), label)

            # Backward
            d_loss_fake.backward()

            d_loss = d_loss_real + d_loss_fake
            D_optimizer.step()

            # ====================================== TRAIN G ============================== #
            G.zero_grad()

            # Get D output for fake images & same dog labels
            d_out_fake = D(fake_images, dog_labels)

            # Compute G loss with fake images & dog_labels
            label.fill_(args.real_label_value)
            g_loss = criterion(torch.sigmoid(d_out_fake), label)

            g_loss.backward()

            G_optimizer.step()

            gen_losses.append(g_loss.item())
            dis_losses.append(d_loss.item())

            lr_schedulerG.step(epoch)
            lr_schedulerD.step(epoch)


    return G, gen_losses, dis_losses

if __name__ == '__main__':

    args = parameters.get_parameters()
    print(args.dataroot)

    seed = 1234
    utils.seed_everything(seed)

    gen_model, gen_losses, dis_losses = train_model(args)

    gen_model_dir = Path(args.save_model_dir)
    gen_model_dir.mkdir(exist_ok=True)
    torch.save(gen_model.state_dict(), os.path.join(args.save_model_dir, 'gen_model.pth'))

