import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.generator import UNetGenerator
from models.discriminator import PatchGANDiscriminator
from models.losses import GANLoss, l1_loss
from data.dataloader import create_dataloader
from utlis import save_checkpoint, load_checkpoint, set_seed, progress_bar

def train(
    data_path,
    epochs=20,               # più epoche per addestramento completo
    batch_size=16,           # batch size ragionevole
    img_size=128,
    lr=2e-4,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    checkpoint_path="../checkpoints/latest.pth"
):
    # Imposta seed per riproducibilità
    set_seed(42)

    # Carica dataloader
    train_loader = create_dataloader(data_path, split='train', batch_size=batch_size, img_size=img_size)

    # Inizializza modelli
    G = UNetGenerator().to(device)
    D = PatchGANDiscriminator().to(device)

    # Loss functions
    gan_loss = GANLoss()
    l1_criterion = l1_loss

    # Ottimizzatori
    optim_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # Se esiste checkpoint, caricalo
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(G, optim_G, checkpoint_path, device)
        load_checkpoint(D, optim_D, checkpoint_path.replace("latest", "disc"), device)

    for epoch in range(start_epoch, epochs):
        G.train()
        D.train()

        for i, batch in enumerate(train_loader):
            sketches = batch['sketch'].to(device)
            real_photos = batch['photo'].to(device)

            # === Train Discriminator ===
            optim_D.zero_grad()

            fake_photos = G(sketches)
            real_input = torch.cat([sketches, real_photos], dim=1)
            fake_input = torch.cat([sketches, fake_photos.detach()], dim=1)

            d_real = D(real_input)
            d_fake = D(fake_input)

            loss_D_real = gan_loss(d_real, True)
            loss_D_fake = gan_loss(d_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optim_D.step()

            # === Train Generator ===
            optim_G.zero_grad()

            fake_input = torch.cat([sketches, fake_photos], dim=1)
            d_fake_for_g = D(fake_input)

            loss_G_gan = gan_loss(d_fake_for_g, True)
            loss_G_l1 = l1_criterion(fake_photos, real_photos) * 100

            loss_G = loss_G_gan + loss_G_l1
            loss_G.backward()
            optim_G.step()

            progress_bar(i + 1, len(train_loader))

        print(f"Epoch [{epoch + 1}/{epochs}] | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")

        # Salva checkpoint ogni epoca
        save_checkpoint(G, optim_G, epoch + 1, checkpoint_path)
        save_checkpoint(D, optim_D, epoch + 1, checkpoint_path.replace("latest", "disc"))

        # Salva immagine esempio generata
        G.eval()
        with torch.no_grad():
            sample_fake = G(sketches[:4])
            sample_grid = torch.cat([sketches[:4], sample_fake], dim=0)
            os.makedirs("results", exist_ok=True)
            save_image((sample_grid * 0.5 + 0.5), f"results/sample_epoch_{epoch + 1}.png", nrow=4)

if __name__ == "__main__":
    data_root = "../data/processed"
    train(data_root, epochs=20, batch_size=16)
