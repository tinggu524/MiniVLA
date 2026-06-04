import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from minivla_data import MiniVLADataset, VOCAB
from minivla_model import MiniVLA

def main():
    Path("checkpoints").mkdir(exist_ok=True)

    dataset = MiniVLADataset("data/demos_v0.npz")

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)

    model = MiniVLA(vocab_size=len(VOCAB)).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            image = batch["image"].to(device)
            state = batch["state"].to(device)
            instruction = batch["instruction"].to(device)
            action = batch["action"].to(device)

            pred_action = model(image, state, instruction)
            loss = loss_fn(pred_action, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print("epoch:", epoch, "loss:", avg_loss)

    torch.save(model.state_dict(), "checkpoints/minivla_v0.pt")
    print("saved checkpoints/minivla_v0.pt")

if __name__ == "__main__":
    main()
