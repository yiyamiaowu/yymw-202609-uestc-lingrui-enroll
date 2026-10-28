import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import lpips

from dataset import DerainDataset
from utils import calculate_metrics, save_checkpoint, load_checkpoint, save_some_examples
from models.baseline_net import BaselineNet
from models.unet import UNet
from models.DerainNet import DerainNet
from models.Mymodel import SPANet
from losses.perceptual_loss import PerceptualLoss
from losses.easy_contrastive_loss import EasyContrastiveLoss
from losses.infoNCE_loss import infoNCEloss

# 把SPANet解包只返回去雨图
class InferenceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, tuple):
            return out[0]
        return out

def get_args():
    parser = argparse.ArgumentParser(description="Deraining Model Training")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "unet", "derainnet", "mymodel"], help="Model to use")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Train or test mode")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for testing")

    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    parser.add_argument("--use_perceptual_loss", action="store_true", help="Use perceptual loss")
    parser.add_argument("--lambda_pixel", type=float, default=1.0, help="Weight for pixel loss")
    parser.add_argument("--lambda_perceptual", type=float, default=0.1, help="Weight for perceptual loss")

    parser.add_argument("--use_easy_contrastive_loss", action="store_true", help="Use easy contrastive loss")
    parser.add_argument("--lambda_easy_contrastive", type=float, default=0.1, help="Weight for easy contrastive loss")

    parser.add_argument("--use_infoNCE_loss", action="store_true", help="Use InfoNCE loss")
    parser.add_argument("--lambda_infoNCE_loss", type=float, default=0.1, help="Weight for infoNCE Loss")

    parser.add_argument("--use_attention_loss", action="store_true")
    parser.add_argument("--lambda_attention", type=float, default=0.01)

    return parser.parse_args()


def train_one_epoch(loader, model, optimizer, pixel_loss_fn, perceptual_loss_fn, contrastive_loss_fn,  infoNCE_loss_fn,args, device):
    loop = tqdm(loader, leave=True)
    model.train()

    for _, (rainy, clean, _) in enumerate(loop):
        rainy, clean = rainy.to(device), clean.to(device)

        output = model(rainy)
        if isinstance(output, tuple):
            derained, attention_maps = output
        else:
            derained, attention_maps = output, None

        postfix = {}

        pixel_loss = args.lambda_pixel * pixel_loss_fn(derained, clean)
        total_loss = pixel_loss
        postfix["pixel"] = f"{pixel_loss.item():.4f}"

        if args.use_perceptual_loss and perceptual_loss_fn is not None:
            p_loss = args.lambda_perceptual * perceptual_loss_fn(derained, clean)
            total_loss = total_loss + p_loss
            postfix["perc"] = f"{p_loss.item():.4f}"

        if args.use_easy_contrastive_loss and contrastive_loss_fn is not None:
            c_loss = args.lambda_easy_contrastive * contrastive_loss_fn(
            derained, rainy, clean)
            total_loss = total_loss + c_loss
            postfix["ctr"] = f"{c_loss.item():.4f}"

        if args.use_infoNCE_loss and infoNCE_loss_fn is not None:
            c_loss = args.lambda_infoNCE_loss * infoNCE_loss_fn(
            derained, rainy, clean)
            total_loss = total_loss + c_loss
            postfix["NCE"] = f"{c_loss.item():.4f}"

        if attention_maps is not None and args.use_attention_loss:
            att_loss = model.compute_attention_loss(attention_maps, rainy, clean)
            if att_loss is not None:
                total_loss = total_loss + args.lambda_attention * att_loss
                postfix["att"] = f"{att_loss.item():.4f}"

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        postfix["total"] = f"{total_loss.item():.4f}"
        loop.set_postfix(**postfix)

def evaluate(loader, model, device, lpips_fn):
    model.eval()
    total_psnr, total_ssim, total_lpips = 0, 0, 0

    with torch.no_grad():
        for rainy, clean, _ in loader:
            rainy, clean = rainy.to(device), clean.to(device)
            out = model(rainy)
            derained = out[0] if isinstance(out, tuple) else out

            psnr, ssim, lpips_score = calculate_metrics(derained, clean, lpips_fn)
            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips_score

    avg_psnr = total_psnr / len(loader)
    avg_ssim = total_ssim / len(loader)
    avg_lpips = total_lpips / len(loader)

    print(f"EVAL => Avg PSNR: {avg_psnr:.4f} | Avg SSIM: {avg_ssim:.4f} | Avg LPIPS: {avg_lpips:.4f}")
    return avg_psnr


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.model == "baseline":
        model = BaselineNet().to(device)
    elif args.model == "unet":
        model = UNet().to(device)
    elif args.model == "derainnet":
        model = DerainNet().to(device)
    elif args.model == "mymodel":
        model = SPANet().to(device)



    lpips_fn = lpips.LPIPS(net='alex').to(device)

    pixel_loss_fn = nn.L1Loss()
    perceptual_loss_fn = PerceptualLoss().to(device) if args.use_perceptual_loss else None
    contrastive_loss_fn = EasyContrastiveLoss().to(device) if args.use_easy_contrastive_loss else None
    infonce_loss_fn = infoNCEloss(tem=0.1).to(device) if args.use_infoNCE_loss else None
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.mode == "train":
        train_dataset = DerainDataset(root_dir=args.data_dir, is_train=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_dataset = DerainDataset(root_dir=args.data_dir, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        best_psnr = 0.0
        for epoch in range(args.epochs):
            print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
            train_one_epoch(train_loader, model, optimizer, pixel_loss_fn, perceptual_loss_fn, contrastive_loss_fn, infonce_loss_fn, args, device)

            current_psnr = evaluate(test_loader, model, device, lpips_fn)

            if current_psnr > best_psnr:
                best_psnr = current_psnr
                checkpoint_data = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                suffix = []
                if args.use_perceptual_loss:
                    suffix.append("perc")
                if args.use_attention_loss:
                    suffix.append("attn")
                if args.use_easy_contrastive_loss:
                    suffix.append("ctr")
                if args.use_infoNCE_loss:
                    suffix.append("nce")
                suffix_str = "_" + "_".join(suffix) if suffix else ""
                model_name = f"{args.model}{suffix_str}.pth.tar"
                save_checkpoint(checkpoint_data, filename=f"best_{model_name}")

            save_some_examples(InferenceWrapper(model), test_loader, epoch, folder="evaluation_images", device=device)

    elif args.mode == "test":
        if not args.checkpoint: raise ValueError("Must provide a checkpoint for test mode.")
        load_checkpoint(torch.load(args.checkpoint, map_location=device), model, optimizer)
        test_dataset = DerainDataset(root_dir=args.data_dir, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        evaluate(test_loader, model, device, lpips_fn)


if __name__ == "__main__":
    if not os.path.exists("checkpoints"): os.makedirs("checkpoints")
    main()