import torch.nn as nn
import torch

class MiniVLA(nn.Module):
    def __init__(self, vocab_size, action_dim=3):
        super().__init__()

        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
        )

        self.language_embedding = nn.Embedding(vocab_size, 32, padding_idx=0)

        self.language_encoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(128 + 64 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.action_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, image, state, instruction):
        vision_feat = self.vision_encoder(image)

        word_feat = self.language_embedding(instruction)
        lang_feat = word_feat.mean(dim=1)
        lang_feat = self.language_encoder(lang_feat)

        state_feat = self.state_encoder(state)

        fused = torch.cat([vision_feat, lang_feat, state_feat], dim=1)
        fused = self.fusion(fused)

        action = self.action_head(fused)
        return action