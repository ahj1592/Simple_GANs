import os

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


from Generator import Generator
from Discriminator import Discriminator
from utils.save_img import save_tensor_images
from utils.noise import get_noise
from utils.loss import get_gen_loss, get_disc_loss

criterion = nn.BCEWithLogitsLoss()
n_epochs = 2000
z_dim = 64
batch_size = 200
lr = 0.00001


dataloader = DataLoader(
    MNIST('data', download=True, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True
)


device = torch.device('cuda')
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)



save_step = len(dataloader) // batch_size
mean_generator_loss = 0
mean_discriminator_loss = 0
test_generator = True # Whether the generator should be tested
gen_loss = False
error = False


base_dir = 'results'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
    print(f'Create the directory {base_dir}')

    
for epoch in range(n_epochs):
    step = 1
    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)

        # Flatten the batch of real images from the dataset
        real = real.view(cur_batch_size, -1).to(device)

        ### Update discriminator ###
        # Zero out the gradients before backpropagation
        disc_opt.zero_grad()

        # Calculate discriminator loss
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)

        # Update gradients
        disc_loss.backward(retain_graph=True)

        # Update optimizer
        disc_opt.step()

        gen_opt.zero_grad()
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
        gen_loss.backward()
        gen_opt.step()


        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item()

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item()

        ### Visualization code ###            
        if step == save_step and (epoch + 1) % 10 == 0:
            print('Save the image')
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            real_image_path = base_dir + '/real_epoch_' + str(epoch + 1) + '.png'
            fake_image_path = base_dir + '/fake_epoch_' + str(epoch + 1) + '.png'
            
            save_tensor_images(fake, img_path=fake_image_path)
            save_tensor_images(real, img_path=real_image_path)
        step += 1
       
    mean_generator_loss /= len(dataloader)
    mean_discriminator_loss /= len(dataloader)
    print(f"epoch {epoch + 1}: Generator loss: {mean_generator_loss}, discriminator loss:{mean_discriminator_loss}")
    mean_generator_loss = 0
    mean_discriminator_loss = 0