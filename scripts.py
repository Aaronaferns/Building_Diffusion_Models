

import os

import torch
import torch.nn.functional as F
from itertools import cycle
from diffusion import diffusionTrainStep, sample
from EMA import EMA
from tqdm import tqdm
import matplotlib.pyplot as plt






def saveCheckPoint(save_folder, exp_no, step, model, optimizer, loss_list, ema):
    save_dir = save_folder
    exp_dir = os.path.join(save_dir, str(exp_no))
    os.makedirs(exp_dir, exist_ok=True)
    ckpt_path = os.path.join(exp_dir, f"{step}_checkpoint.pt")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "losses": loss_list,
        "ema": ema.state_dict()
    }, ckpt_path)



def train(
    model,
    dataLoader,
    optimizer,
    loss_fn,
    num_steps,
    save_interval,
    save_folder,
    exp_no,
    device,
    sample_interval,
    eval_interval,
    ema_decay=0.9999,
    fid_fn=None,                 # callable: fid_fn(model)->float
    sample_save_fn=None          # callable: sample_save_fn(model, step)->None
    ):
    
    model.train()
    data_iter = cycle(dataLoader)
    pbar = tqdm(range(num_steps), desc="Training")
    ema = EMA(model, decay=ema_decay)
    loss_list = []
    for step in pbar:
        X0, _ = next(data_iter)
        loss = diffusionTrainStep(model, optimizer, loss_fn, X0, device)
        loss_list.append(loss)
        
        # ----- periodic sampling (EMA weights) -----
        if sample_save_fn is not None and (step > 0 and step % sample_interval == 0):
            model.eval()
            ema.apply_to(model)
            try:
                sample_save_fn(model, step)
            finally:
                ema.restore(model)
                model.train()

        # ----- periodic FID (EMA weights) -----
        if fid_fn is not None and (step > 0 and step % eval_interval == 0):
            model.eval()
            ema.apply_to(model)
            try:
                fid_val = fid_fn(model)
                print(f"\n[step {step}] FID(EMA) = {fid_val:.3f}")
            finally:
                ema.restore(model)
                model.train()

        # ----- checkpoint -----
        if (step > 0 and step % save_interval == 0) or (step == num_steps - 1):
            saveCheckPoint(
                save_folder=save_folder,
                exp_no=exp_no,
                step=step,
                model=model,
                optimizer=optimizer,
                loss_list=loss_list
                ema = ema
            )
    
    print("Done Training")
    return loss_list
    

