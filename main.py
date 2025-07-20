import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from dataset import download_enwik8, ByteSequenceDataset
from model import Encoder, Decoder

def cosine_loss(h_pred, h_true):
    return (1.0 - (h_pred * h_true).sum(dim=-1)).mean()

def train_model():
    wandb.init(project="enwik8-per-step-model", config={
        "state_dim": 128,
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
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    encoder = Encoder(cfg.batch_size, cfg.state_dim).to(device)
    decoder = Decoder(cfg.batch_size, cfg.state_dim).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg.lr)

    global_step = 0
    for epoch in range(cfg.epochs):
        for batch in loader:
            x_seq = batch.to(device)
            h = torch.zeros((cfg.batch_size, cfg.state_dim), device=device)
            h[:, 0] = 1.0  # Initialize first state to unit vector
            h += 0.01 * torch.randn_like(h)  # Add small noise to initial state
            h = F.normalize(h, dim=-1)  # Normalize initial state to unit norm
            for t in range(x_seq.size(1)):
                x_t = x_seq[:, t]
                h_next = encoder(h, x_t)
                #add small noise to h_next
                h_next = h_next + 0.01 * torch.randn_like(h_next)
                #normalize h_next to unit norm
                h_next = F.normalize(h_next, dim=-1)
                h_recon, logits = decoder(h_next)

                loss_h = cosine_loss(h_recon, h)
                loss_x = F.cross_entropy(logits, x_t)
                loss = loss_h + loss_x

                opt.zero_grad()
                loss.backward()
                opt.step()

                if global_step % cfg.log_interval == 0:
                    wandb.log({
                        "loss": loss.item(),
                        "cosine_loss": loss_h.item(),
                        "ce_x": loss_x.item(),
                        "mean_state_norm": h_next.norm(dim=-1).mean().item(),
                        "mean_state_std": h.std(dim=0).mean().item(),
                        "delta_norm": (h_next - h).norm(dim=-1).mean().item(),
                        "h_variance": h.var(dim=0).mean().item(),
                        "mean_cos_sim": (F.normalize(h_recon, dim=-1) * h).sum(dim=-1).mean().item(),
                        "step": global_step,
                    })
                    print(f"\rEpoch {epoch+1}/{cfg.epochs}, Step {global_step}, Loss: {loss.item():.4f}", end='', flush=True)

                h = h_next.detach()
                global_step += 1
            with torch.no_grad():
                h_seq = [torch.zeros_like(h) for _ in range(x_seq.size(1) + 1)]
                h_seq[0][:, 0] = 1.0  # initial h_0
                for t in range(x_seq.size(1)):
                    h_seq[t + 1] = encoder(h_seq[t], x_seq[:, t])

                h_rev = h_seq[-1]
                x_correct = [] #binary if reconstruction is correct
                first_faulty = [x_seq.size(1) for _ in range(cfg.batch_size)]
                for t in reversed(range(x_seq.size(1))):
                    correct_token = x_seq[:, t] # ground truth token for timestep t
                    h_recon, logits = decoder(h_rev)
                    h_rev = h_recon
                    pred_token = logits.argmax(dim=-1)
                    is_correct = (pred_token == correct_token)

                    for i in range(cfg.batch_size):
                        if not is_correct[i] and first_faulty[i] == x_seq.size(1):
                            first_faulty[i] = x_seq.size(1) - t  # record first faulty step
                    x_correct.append(is_correct.float())

                x_correct = torch.stack(list(reversed(x_correct)), dim=1)  # shape (B, T)
                accuracy = x_correct.sum().item() / cfg.batch_size  # mean number of correct tokens per seq

                # Fill in full length for perfect sequences
                mean_first_error = sum(first_faulty) / len(first_faulty)

                wandb.log({
                    "reverse_reconstruction_accuracy": accuracy,
                    "mean_steps_to_first_error": mean_first_error,
                    "step": global_step
                })
                print(
                    f"\nEpoch {epoch + 1}/{cfg.epochs}, Reverse Accuracy: {accuracy:.4f}, Mean Steps to First Error: {mean_first_error:.2f}")


if __name__ == "__main__":
    train_model()
