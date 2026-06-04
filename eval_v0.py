import numpy as np
import torch
import imageio.v2 as imageio
import json
from pathlib import Path

from minivla_data import render_scene, encode_instruction, VOCAB
from minivla_model import MiniVLA

def rollout(model, device, save_gif=False, gif_path="outputs/v0/eval_rollout_v0.gif"):
    image_size = 64
    max_steps = 40
    instruction = "move red block to green target"

    block_pos = np.random.uniform(10, 54, size=2)
    target_pos = np.random.uniform(10, 54, size=2)
    gripper_pos = np.random.uniform(10, 54, size=2)
    is_holding = False

    frames = []

    for step in range(max_steps):
        image = render_scene(block_pos, target_pos, gripper_pos, image_size)

        frames.append(image)

        image_tensor = torch.tensor(
            np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))
        ).unsqueeze(0).to(device)

        state = np.array([
            gripper_pos[0] / image_size,
            gripper_pos[1] / image_size,
            float(is_holding),
        ], dtype=np.float32)

        state_tensor = torch.tensor(state).unsqueeze(0).to(device)

        instruction_ids = encode_instruction(instruction)
        instruction_tensor = torch.tensor(instruction_ids).unsqueeze(0).to(device)

        with torch.no_grad():
            action = model(image_tensor, state_tensor, instruction_tensor)

        action = action.squeeze(0).cpu().numpy()

        gripper_pos = gripper_pos + action[:2]
        gripper_pos = np.clip(gripper_pos, 0, image_size - 1)

        if action[2] > 0.5:
            is_holding = True

        if is_holding:
            block_pos = gripper_pos.copy()

        if action[2] < -0.5:
            is_holding = False

    final_distance = np.linalg.norm(block_pos - target_pos)
    success = final_distance < 5.0

    if save_gif:
        imageio.mimsave(gif_path, frames, duration=0.2)

    return success, final_distance, frames


def main():
    Path("outputs/v0").mkdir(parents=True, exist_ok=True)
    failure_dir = Path("outputs/v0/rollouts/failures")
    success_dir = Path("outputs/v0/rollouts/success_examples")
    failure_dir.mkdir(parents=True, exist_ok=True)
    success_dir.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)

    model = MiniVLA(vocab_size=len(VOCAB)).to(device)
    model.load_state_dict(torch.load("checkpoints/minivla_v0.pt", map_location=device))
    model.eval()

    num_trials = 100
    max_success_gifs = 5
    successes = 0
    distances = []
    failure_trials = []
    success_gifs_saved = 0

    for i in range(num_trials):
        success, final_distance, frames = rollout(
            model,
            device,
            save_gif=False,
        )

        successes += int(success)
        distances.append(final_distance)

        if success and success_gifs_saved < max_success_gifs:
            gif_path = success_dir / f"trial_{i:03d}_distance_{final_distance:.2f}.gif"
            imageio.mimsave(gif_path, frames, duration=0.2)
            success_gifs_saved += 1

        if not success:
            failure_trials.append({
                "trial": i,
                "final_distance": float(final_distance),
            })
            gif_path = failure_dir / f"trial_{i:03d}_distance_{final_distance:.2f}.gif"
            imageio.mimsave(gif_path, frames, duration=0.2)

        if i % 10 == 0:
            print("trial:", i, "success:", success, "distance:", final_distance)

    success_rate = successes / num_trials
    mean_distance = np.mean(distances)

    print("success rate:", success_rate)
    print("mean final distance:", mean_distance)
    print("failure trials:", failure_trials)
    print("saved failure gifs to outputs/v0/rollouts/failures")
    print("saved success examples to outputs/v0/rollouts/success_examples")

    results = {
        "model": "MiniVLA-v0",
        "num_trials": num_trials,
        "success_rate": success_rate,
        "mean_final_distance": float(mean_distance),
        "failure_trials": failure_trials,
        "failure_gif_dir": "outputs/v0/rollouts/failures",
        "success_example_gif_dir": "outputs/v0/rollouts/success_examples",
        "max_success_gifs": max_success_gifs,
    }

    with open("outputs/v0/results_v0.json", "w") as f:
        json.dump(results, f, indent=2)

    print("saved outputs/v0/results_v0.json")

if __name__ == "__main__":
    main()
