import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from dataset_rrt import DatasetRRT
from env import Env
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm.auto import tqdm
from image import Image

from diffusers import DDPMScheduler, UNet1DModel
from diffusers import get_cosine_schedule_with_warmup
from diffusers.utils.torch_utils import randn_tensor


def plot_paths(filename, env: Env, paths: np.ndarray):
    print(f"PATH RESULTS: {filename} : {len(paths)}")
    for path in paths:
        print(path.T)

    image = Image(filename, env)

    for path in paths:
        image.add_path(path.T, color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))


@dataclass
class TrainingConfig:
    sequence_length = 32
    in_channels = 2
    out_channels = 2
    train_batch_size = 64
    eval_batch_size = 128
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_output_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"
    inpaint = True
    seed = 0


def steer_vector(point: torch.Tensor, env: Env):
    env_point = tuple(point.tolist())
    if env.point_collides(env_point):
        return torch.tensor(env.nearest_free_point(env_point)) - point
    else:
        return point - torch.tensor(env.nearest_collision_point(env_point))


def evaluate(
        unet,
        scheduler,
        noise,
        generator,
        device,
        env: Env,
        num_inference_steps: int = 100,
        inpaint_mask=None,
        steer_length=False,
        steer_obstacles=False,
) -> torch.Tensor:
    model.eval()
    scheduler.set_timesteps(num_inference_steps, device=device)

    sequence = noise.clone().detach()
    if inpaint_mask is not None:
        sequence[~inpaint_mask.isnan()] = inpaint_mask[~inpaint_mask.isnan()]

    for t in tqdm(scheduler.timesteps):
        model_output = unet(sequence, t).sample

        # denoise
        sequence = scheduler.step(
            model_output, t, sequence, generator=generator
        ).prev_sample

        # inpaint
        if inpaint_mask is not None:
            sequence[~inpaint_mask.isnan()] = inpaint_mask[~inpaint_mask.isnan()]

        # steer
        if t < 30:
            if steer_obstacles:
                steer_vectors = torch.empty_like(sequence)
                for i in range(sequence.shape[0]):
                    for j in range(sequence.shape[2]):
                        steer_vectors[i, :, j] = steer_vector(sequence[i, :, j], env)
                sequence = sequence + 0.1 * steer_vectors

            if steer_length:
                distances = (sequence[:, :, :-1] - sequence[:, :, 1:]) ** 2
                sequence = (
                        sequence - 0.1 * torch.autograd.grad(distances.sum(), sequence)[0]
                )

        # inpaint
        if inpaint_mask is not None:
            sequence[~inpaint_mask.isnan()] = inpaint_mask[~inpaint_mask.isnan()]
    return sequence


def train_loop(
        config: TrainingConfig,
        model,
        noise_scheduler,
        optimizer,
        train_dataloader,
        lr_scheduler,
):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    device = accelerator.device
    model.to(device)

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        model.train()
        for step, batch in enumerate(train_dataloader):
            observation = batch[:, :2, :]

            clean_images = observation
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (clean_images.shape[0],),
                device=clean_images.device,
            ).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        model.eval()
        seed_generator = torch.manual_seed(config.seed)
        tst_shape = (30, model.config.in_channels, model.config.sample_size)
        tst_input = torch.full(tst_shape, float("Nan"), device=model.device)
        tst_input[:, :, 0] = 0.5
        angles = torch.linspace(0, 2 * (1 - 1 / 30) * np.pi, 30, dtype=torch.float32)
        goal = np.stack(
            [0.5 + 0.5 * np.cos(angles), 0.5 + 0.5 * np.sin(angles)], axis=1
        )
        tst_input[:, :, -1] = torch.tensor(goal)
        tst_denoised = evaluate(
            model,
            scheduler=noise_scheduler,
            noise=torch.randn(tst_shape, device=model.device),
            generator=seed_generator,
            device=model.device,
            env=env,
            num_inference_steps=50,
            inpaint_mask=tst_input,
        )
        collisions = 0
        for i in range(tst_denoised.shape[0]):
            for j in range(tst_denoised.shape[1] - 1):
                collisions += env.segment_collides(
                    tuple(tst_denoised[i, :, j].tolist()),
                    tuple(tst_denoised[i, :, j + 1].tolist()),
                )
        collision_rate = collisions / (
                tst_denoised.shape[0] * (tst_denoised.shape[1] - 1)
        )
        logs = {
            "loss": loss.detach().item(),
            "lr": lr_scheduler.get_last_lr()[0],
            "step": global_step,
            "collision_rate": collision_rate,
        }
        progress_bar.set_postfix(**logs)
    accelerator.end_training()


def evaluate_steering_combinations(env: Env, inpaint_mask=None):
    tst_shape = (
        (10, model.config.in_channels, model.config.sample_size)
        if inpaint_mask is None
        else inpaint_mask.shape
    )
    seed_generator = torch.manual_seed(config.seed)
    tst_input = randn_tensor(tst_shape, generator=seed_generator, device=model.device)
    plot_paths("../out/noise_input.svg", env, tst_input.cpu().detach().numpy())
    tst_denoised = evaluate(
        model,
        scheduler=noise_scheduler,
        noise=tst_input,
        generator=seed_generator,
        device=model.device,
        env=env,
        num_inference_steps=300,
        steer_length=False,
        steer_obstacles=False,
        inpaint_mask=inpaint_mask,
    )
    plot_paths("../out/easy_no_steering.svg", env, tst_denoised.cpu().detach().numpy())

    seed_generator = torch.manual_seed(config.seed)
    tst_input = randn_tensor(tst_shape, generator=seed_generator, device=model.device)
    tst_denoised = evaluate(
        model,
        scheduler=noise_scheduler,
        noise=tst_input,
        generator=seed_generator,
        device=model.device,
        env=env,
        num_inference_steps=300,
        steer_length=True,
        steer_obstacles=False,
        inpaint_mask=inpaint_mask,
    )
    plot_paths("../out/easy_length_steering.svg", env, tst_denoised.cpu().detach().numpy())

    seed_generator = torch.manual_seed(config.seed)
    tst_input = randn_tensor(tst_shape, generator=seed_generator, device=model.device)
    tst_denoised = evaluate(
        model,
        scheduler=noise_scheduler,
        noise=tst_input,
        generator=seed_generator,
        device=model.device,
        env=env,
        num_inference_steps=300,
        steer_length=False,
        steer_obstacles=True,
        inpaint_mask=inpaint_mask,
    )
    plot_paths("../out/easy_obstacle_steering.svg", env, tst_denoised.cpu().detach().numpy())

    seed_generator = torch.manual_seed(config.seed)
    tst_input = randn_tensor(tst_shape, generator=seed_generator, device=model.device)
    tst_denoised = evaluate(
        model,
        scheduler=noise_scheduler,
        noise=tst_input,
        generator=seed_generator,
        device=model.device,
        env=env,
        num_inference_steps=300,
        steer_length=True,
        steer_obstacles=True,
        inpaint_mask=inpaint_mask,
    )
    plot_paths("../out/easy_obstacle_and_length_steering.svg", env, tst_denoised.cpu().detach().numpy())


if __name__ == "__main__":
    env = Env(1.0, 1.0)
    env.add_obstacle([(0.32, 0.37), (0.29, 0.56), (0.42, 0.68), (0.58, 0.69), (0.70, 0.52), (0.66, 0.35), (0.49, 0.29)])

    config = TrainingConfig()

    dataset_trn = DatasetRRT("../datasets/datasetRRT_easy.pt")
    print(f"Episodes in train dataset: {len(dataset_trn)}")

    dataloader_trn = DataLoader(
        dataset_trn, batch_size=config.train_batch_size, shuffle=True, num_workers=1
    )

    model = UNet1DModel(
        sample_size=config.sequence_length,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        time_embedding_type="positional",
        use_timestep_embedding=True,
        act_fn="mish",
        norm_num_groups=1,
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(
            config.in_channels * 8,
            config.in_channels * 16,
            config.in_channels * 32,
        ),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock1D",
            "DownBlock1D",
            "DownBlock1D",
        ),
        mid_block_type="UNetMidBlock1D",
        up_block_types=(
            "UpBlock1D",
            "UpBlock1D",
            "UpBlock1D",
        ),
        out_block_type="OutConv1DBlock",
        add_attention=False,
        dropout=0.0,
    )

    summary(
        model,
        (config.train_batch_size, config.in_channels, config.sequence_length),
        timestep=0,
        depth=4,
        col_names=["input_size", "output_size", "num_params"],
        row_settings=["hide_recursive_layers"],
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=300)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(dataloader_trn) * config.num_epochs),
    )

    time_start = time.time()
    train_loop(config, model, noise_scheduler, optimizer, dataloader_trn, lr_scheduler)
    print("\n\n")
    print(f"Training time: {time.time() - time_start}")

    n = 10
    inpaint_shape = (n, config.in_channels, config.sequence_length)
    inpaint_mask = torch.full(inpaint_shape, float('Nan'), device=model.device)
    inpaint_mask[:, 0, 0] = 0.1
    inpaint_mask[:, 1, 0] = 0.5
    inpaint_mask[:, 0, -1] = 0.9
    inpaint_mask[:, 1, -1] = 0.5
    evaluate_steering_combinations(env, inpaint_mask=inpaint_mask)
