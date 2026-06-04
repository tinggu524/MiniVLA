import numpy as np
import imageio.v2 as imageio
import torch
from torch.utils.data import Dataset
from pathlib import Path

# 2d桌面
def draw_square(image, center, size, color):
    x = int(center[0])
    y = int(center[1])
    half = size // 2

    x0 = max(0, x - half)
    x1 = min(image.shape[1], x + half)
    y0 = max(0, y - half)
    y1 = min(image.shape[0], y + half)

    image[y0:y1, x0:x1] = color

def render_scene(block_pos, target_pos, gripper_pos, image_size=64):
    image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

    draw_square(image, target_pos, size=12, color=(0, 220, 0))
    draw_square(image, block_pos, size=8, color=(220, 0, 0))
    draw_square(image, gripper_pos, size=6, color=(0, 80, 255))

    return image


# 初始移动 expert
def expert_move_toward(current_pos, target_pos, step_size=2.0):
    diff = target_pos - current_pos
    distance = np.linalg.norm(diff)

    if distance < step_size:
        action = diff
    # 向量移动的距离
    else:
        direction = diff / distance
        action = direction * step_size

    return action

def expert_pick_place(gripper_pos, block_pos, target_pos, is_holding):
    if not is_holding:
        distance_to_block = np.linalg.norm(block_pos - gripper_pos)

        if distance_to_block < 3.0:
            move_action = np.array([0.0, 0.0])
            gripper_action = 1.0
        else:
            move_action = expert_move_toward(gripper_pos, block_pos)
            gripper_action = 0.0

    else:
        distance_to_target = np.linalg.norm(target_pos - gripper_pos)

        if distance_to_target < 3.0:
            move_action = np.array([0.0, 0.0])
            gripper_action = -1.0
        else:
            move_action = expert_move_toward(gripper_pos, target_pos)
            gripper_action = 1.0

    return np.array([move_action[0], move_action[1], gripper_action])

# 生成数据
def generate_episode(image_size=64, max_steps=40):
    block_pos = np.random.uniform(10, 54, size=2)
    target_pos = np.random.uniform(10, 54, size=2)
    gripper_pos = np.random.uniform(10, 54, size=2)

    is_holding = False
    instruction = "move red block to green target"

    images = []
    states = []
    actions = []
    instructions = []
    frames = []

    for step in range(max_steps):
        image = render_scene(block_pos, target_pos, gripper_pos, image_size)
        frames.append(image)

        state = np.array([
            gripper_pos[0] / image_size,
            gripper_pos[1] / image_size,
            float(is_holding),
        ], dtype=np.float32)

        action = expert_pick_place(gripper_pos, block_pos, target_pos, is_holding)

        images.append(image)
        states.append(state)
        actions.append(action.astype(np.float32))
        instructions.append(instruction)

        gripper_pos = gripper_pos + action[:2]

        if action[2] > 0.5:
            is_holding = True

        if is_holding:
            block_pos = gripper_pos.copy()

        if action[2] < -0.5:
            is_holding = False

    return {
        "images": images,
        "states": states,
        "actions": actions,
        "instructions": instructions,
        "frames": frames,
    }

VOCAB = {
    "<pad>": 0,
    "move": 1,
    "red": 2,
    "block": 3,
    "to": 4,
    "green": 5,
    "target": 6,
}

def encode_instruction(instruction, max_len=8):
    words = instruction.split()
    ids = [VOCAB[word] for word in words]

    while len(ids) < max_len:
        ids.append(VOCAB["<pad>"])

    return np.array(ids[:max_len], dtype=np.int64)

class MiniVLADataset(Dataset):
    def __init__(self, path):
        data = np.load(path)

        self.images = data["images"].astype(np.float32) / 255.0
        self.states = data["states"].astype(np.float32)
        self.actions = data["actions"].astype(np.float32)
        self.instructions = data["instructions"]

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = np.transpose(image, (2, 0, 1))

        state = self.states[idx]
        action = self.actions[idx]
        instruction = encode_instruction(str(self.instructions[idx]))

        return {
            "image": torch.tensor(image),
            "state": torch.tensor(state),
            "instruction": torch.tensor(instruction),
            "action": torch.tensor(action),
        }

def main():
    image_size = 64
    num_episodes = 500

    Path("data").mkdir(exist_ok=True)
    Path("outputs/v0").mkdir(parents=True, exist_ok=True)

    all_images = []
    all_states = []
    all_actions = []
    all_instructions = []

    example_frames = None

    for episode_idx in range(num_episodes):
        episode = generate_episode(image_size=image_size)

        all_images.extend(episode["images"])
        all_states.extend(episode["states"])
        all_actions.extend(episode["actions"])
        all_instructions.extend(episode["instructions"])

        if episode_idx == 0:
            example_frames = episode["frames"]

        if episode_idx % 50 == 0:
            print("generated episode:", episode_idx)

    np.savez(
        "data/demos_v0.npz",
        images=np.array(all_images),
        states=np.array(all_states),
        actions=np.array(all_actions),
        instructions=np.array(all_instructions),
    )

    imageio.mimsave("outputs/v0/example_episode.gif", example_frames, duration=0.2)

    data = np.load("data/demos_v0.npz")
    print("saved data/demos_v0.npz")
    print("images shape:", data["images"].shape)
    print("states shape:", data["states"].shape)
    print("actions shape:", data["actions"].shape)
    print("instructions shape:", data["instructions"].shape)
    print("saved outputs/v0/example_episode.gif")

    dataset = MiniVLADataset("data/demos_v0.npz")
    sample = dataset[0]

    print("dataset size:", len(dataset))
    print("sample image:", sample["image"].shape)
    print("sample state:", sample["state"])
    print("sample instruction:", sample["instruction"])
    print("sample action:", sample["action"])



if __name__ == "__main__":
    main()
