import pickle
from models import Unet
from datasetLoaders import make_cifar10_train_loader
import torch
import matplotlib.pyplot as plt
from scripts import train
import torch.nn.functional as F
from metrics import compute_real_stats

def main():
    
    device = device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    real_mu, real_sigma = compute_real_stats(device=device, n_real=50000, batch_size=128)
    
    model = Unet(
        in_resolution = 32,  # CIFAR-10 images are 32x32
        input_ch = 3,
        ch = 128,
        output_ch = 3,
        num_res_blocks = 3,
        temb_dim = 256,
        attn_res = set([16]),  # Attention at lower resolutions for 32x32
        dropout = 0.1,
        resam_with_conv=True,
        ch_mult=[1,2,4,8]
        )

    model = model.to(device)
    dataLoader = make_cifar10_train_loader()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=0.0 )
    loss_fn = F.mse_loss
    NUM_TRAIN_STEPS = 1000
    exp_no = 1
    save_interval = 100
    save_folder = "saves"
    loss_list = train(model, dataLoader, optimizer, loss_fn, NUM_TRAIN_STEPS, save_interval, save_folder, exp_no)
    
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel("Training step")
    plt.ylabel("MSE loss")
    plt.title("DDPM Training Loss")
    plt.savefig(f"{exp_no}_loss_curve.png", dpi=200, bbox_inches="tight")
    plt.show()


    
    
    
    
    
if __name__ == "__main__":
    main()

