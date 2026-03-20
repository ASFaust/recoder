import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from dataset import download_enwik8, ByteSequenceDataset
from model import Encoder, Decoder

sqrt_beta = 0.01

def train_model():
    wandb.init(project="enwik8-per-step-model", config={
        "state_dim": 256,
        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 3,
        "block_size": 512,
        "log_interval": 100
    })
    cfg = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = download_enwik8()
    dataset = ByteSequenceDataset(data, block_size=cfg.block_size)
    eval_dataset = ByteSequenceDataset(data, block_size=cfg.block_size, eval=True)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    encoder = Encoder(cfg.state_dim).to(device)
    decoder = Decoder(cfg.state_dim).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg.lr)

    global_step = 0
    for epoch in range(cfg.epochs):
        for batch in loader:
            x_seq = batch.to(device)
            h = torch.zeros((cfg.batch_size, cfg.state_dim), device=device)
            h[:, 0] = 1.0 # initial h with first dimension set to 1, rest 0
            for t in range(x_seq.size(1)):
                x_t = x_seq[:, t]

                # encoder output on sphere
                h_next = F.normalize(encoder(h, x_t), dim=-1)

                # tangent space langevin noise
                noise = torch.randn_like(h_next)
                noise = noise - (noise * h_next).sum(dim=-1, keepdim=True) * h_next
                h_next_noised = F.normalize(h_next + sqrt_beta * noise, dim=-1)

                # decoder reconstructs previous state and input
                h_recon, logits = decoder(h_next_noised)
                h_recon = F.normalize(h_recon, dim=-1)

                # cosine loss on sphere, CE loss for bytes
                loss_h = 1 - (h_recon * h.detach()).sum(dim=-1).mean()
                loss_x = F.cross_entropy(logits, x_t)

                # cosine similarity matrix over batch
                S = h_next @ h_next.T  # (B, B)
                # penalize off-diagonal similarities
                mask = 1 - torch.eye(cfg.batch_size, device=device)
                loss_spread = (S * mask).pow(2).mean()

                loss = loss_h + loss_x #+ 0.1 * loss_spread

                opt.zero_grad()
                loss.backward()
                opt.step()

                if global_step % cfg.log_interval == 0:
                    wandb.log({
                        "loss/total": loss.item(),
                        "loss/cos_h": loss_h.item(),
                        "loss/ce_x": loss_x.item(),
                        "loss/spread": loss_spread.item(),
                        "loss/ratio_h_to_x": loss_h.item() / (loss_x.item() + 1e-8),
                        "state/mean_norm": h_next.norm(dim=-1).mean().item(),
                        "state/mean_std": h.std(dim=0).mean().item(),
                        "state/delta_norm": (h_next - h).norm(dim=-1).mean().item(),
                        "state/h_variance": h.var(dim=0).mean().item(),
                        "state/mean_cos_sim": (F.normalize(h_recon, dim=-1) * h).sum(dim=-1).mean().item(),
                        "step": global_step,
                    })
                    print(f"\rEpoch {epoch+1}/{cfg.epochs}, Step {global_step}, Loss: {loss.item():.4f}", end='', flush=True)

                h = h_next.detach()
                global_step += 1
            
            with torch.no_grad():
                h_seq = [torch.zeros_like(h) for _ in range(x_seq.size(1) + 1)]
                # Initialize h_seq[0] with the same initial state as training
                h_seq[0][:, 0] = 1.0
                for t in range(x_seq.size(1)):
                    h_seq[t + 1] = encoder(h_seq[t], x_seq[:, t])
                    h_seq[t + 1] = F.normalize(h_seq[t + 1], dim=-1)  # ensure unit norm

                h_rev = h_seq[-1]
                x_correct = [] #binary if reconstruction is correct
                first_faulty = [x_seq.size(1) for _ in range(cfg.batch_size)]
                for t in reversed(range(x_seq.size(1))):
                    correct_token = x_seq[:, t] # ground truth token for timestep t
                    h_recon, logits = decoder(h_rev)
                    h_recon = F.normalize(h_recon, dim=-1)  # ensure unit norm
                    h_rev = h_recon
                    pred_token = logits.argmax(dim=-1)
                    is_correct = (pred_token == correct_token)

                    for i in range(cfg.batch_size):
                        if not is_correct[i] and first_faulty[i] == x_seq.size(1):
                            first_faulty[i] = x_seq.size(1) - t  # record first faulty step
                    x_correct.append(is_correct.float())

                x_correct = torch.stack(list(reversed(x_correct)), dim=1)  # shape (B, T)
                correct_token_count = x_correct.sum().item() / cfg.batch_size  # mean number of correct tokens per seq

                # Fill in full length for perfect sequences
                mean_first_error = sum(first_faulty) / len(first_faulty)

                wandb.log({
                    "eval/correct_token_count": correct_token_count,
                    "eval/mean_steps_to_first_error": mean_first_error,
                    "step": global_step
                })
                print(
                    f"\nEpoch {epoch + 1}/{cfg.epochs}, Eval Correct Tokens per Seq: {correct_token_count:.2f}, Mean Steps to First Error: {mean_first_error:.2f}"
                )

if __name__ == "__main__":
    train_model()
