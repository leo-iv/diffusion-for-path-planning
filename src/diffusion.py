"""
Training and evaluation of the diffusion model for 2D motion planning task.

The training script and model evaluation is a modification of the training script from master's thesis "Diffusion models for path planning"
written by Petr Zahradník at CTU in Prague.
"""

from dataclasses import dataclass
import torch
import numpy as np
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchinfo import summary
import torch.nn.functional as F
from tqdm.auto import tqdm
import time

from diffusers import DDPMScheduler, UNet1DModel
from diffusers import get_cosine_schedule_with_warmup
from diffusers.utils.torch_utils import randn_tensor

from env import Env
from datasets import PathPlanningDataset

from image import Image


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
    mixed_precision = "fp16"
    seed = 0


def _train_loop(
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

        logs = {
            "loss": loss.detach().item(),
            "lr": lr_scheduler.get_last_lr()[0],
            "step": global_step,
        }
        progress_bar.set_postfix(**logs)
    accelerator.end_training()


def train_model(model_path: str, dataset: PathPlanningDataset, config=TrainingConfig()):
    """
    Trains a diffusion model for a 2D motion planning task and saves it into .pt file.

    Args:
        model_path: model output file (should have .pt extension)
        dataset: PathPlanningDataset with the training data
        config: TrainingConfig dataclass with training parameters
    """
    print(f"Episodes in train dataset: {len(dataset)}")

    dataloader = DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=1
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
        num_training_steps=(len(dataloader) * config.num_epochs),
    )

    time_start = time.time()
    _train_loop(config, model, noise_scheduler, optimizer, dataloader, lr_scheduler)
    print("\n\n")
    print(f"Training time: {time.time() - time_start}")

    torch.save(model.state_dict(), model_path)


def __plot_paths(env, paths, file_path):
    paths = paths.cpu().detach().numpy()
    img = Image(file_path, env)
    img.add_paths(np.transpose(paths, (0, 2, 1)))


def evaluate(model, scheduler, noise, generator, device, env: Env, num_inference_steps: int = 300, inpaint_mask=None,
             steer_length=False, steer_obstacles=False, img_output_dir=None) -> torch.Tensor:
    model.eval()
    scheduler.set_timesteps(num_inference_steps, device=device)

    sequence = noise.clone().detach()

    # input noise image
    if img_output_dir is not None:
        __plot_paths(env, sequence, f"{img_output_dir}/input_noise.svg")

    if inpaint_mask is not None:
        sequence[~inpaint_mask.isnan()] = inpaint_mask[~inpaint_mask.isnan()]

    for t in tqdm(scheduler.timesteps):
        model_output = model(sequence, t).sample

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
                steer_obstacle_vectors = torch.empty_like(sequence)
                for path in range(sequence.shape[0]):
                    for point in range(sequence.shape[2]):
                        env_point = tuple(sequence[path, :, point].tolist())
                        if env.point_collides(env_point):
                            steer_obstacle_vectors[path, :, point] = torch.tensor(env.nearest_free_point(env_point))
                        else:
                            steer_obstacle_vectors[path, :, point] = sequence[path, :,
                                                                     point]  # torch.tensor(env.nearest_collision_point(env_point))

                distances = (sequence - steer_obstacle_vectors) ** 2
                sequence = (sequence - 0.2 * torch.autograd.grad(distances.sum(), sequence)[0])

            if steer_length:
                distances = (sequence[:, :, :-1] - sequence[:, :, 1:]) ** 2
                sequence = (
                        sequence - 0.1 * torch.autograd.grad(distances.sum(), sequence)[0]
                )

        # inpaint
        if inpaint_mask is not None:
            sequence[~inpaint_mask.isnan()] = inpaint_mask[~inpaint_mask.isnan()]

        # create image (for every 30th diffusion step or if t < 30)
        if img_output_dir is not None and (t % 30 == 0 or t < 30):
            __plot_paths(env, sequence, f"{img_output_dir}/t={t}.svg")

    # create result image
    if img_output_dir is not None:
        __plot_paths(env, sequence, f"{img_output_dir}/result.svg")

    return sequence


class PathDenoiser:
    def __init__(self, model_path: str, env: Env, config=TrainingConfig()):
        """
        Args:
            model_path: .pt file generated by train_model function
            env: env.Env instance
            config: TrainingConfig used during training (if it is different from the default one)
        """
        self.env = env
        self.config = config
        self.model = UNet1DModel(
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

        self.model.load_state_dict(torch.load(model_path, weights_only=True))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=300)

    def generate_paths(self, starts, goals, samples_per_start: int, num_inference_steps: int, steer_length=False,
                       steer_obstacles=False, img_output_dir=None):
        """
        Generates n paths using the diffusion model.
        Args:
            starts: numpy array of start configurations - shape: (n, 2)
            goals: numpy array of goal configurations - shape: (n, 2)
            samples_per_start: number of samples to generate per one start-goal configuration
            num_inference_steps: number of diffusion steps used during inference
            steer_length: must be set to True to use length steering during inference
            steer_obstacles: must be set to True to use obstacle steering during inference
            img_output_dir: output directory where svg images of the diffusion process will be stored

        Returns:
            result paths as numpy array of shape (n, path_length, 2)
        """
        n = len(starts) * samples_per_start
        inpaint_shape = (n, self.config.in_channels, self.config.sequence_length)
        inpaint_mask = torch.full(inpaint_shape, float('Nan'), device=self.model.device)
        inpaint_mask[:, :, 0] = torch.tensor(np.tile(starts, (samples_per_start, 1)))
        inpaint_mask[:, :, -1] = torch.tensor(np.tile(goals, (samples_per_start, 1)))

        seed_generator = torch.manual_seed(self.config.seed)
        noise_input = randn_tensor(inpaint_shape, generator=seed_generator, device=self.model.device)
        diffusion_result = evaluate(self.model, self.noise_scheduler, noise_input, seed_generator, self.model.device,
                                    self.env, num_inference_steps, inpaint_mask, steer_length, steer_obstacles,
                                    img_output_dir)
        paths = diffusion_result.cpu().detach().numpy()

        return np.transpose(paths, (0, 2, 1))  # reshaping paths from (n, 2, length) to (n, length, 2)

    def generate_random_paths(self, n: int, num_inference_steps: int, steer_length=False, steer_obstacles=False,
                              img_output_dir=None):
        """
        Generates n paths using the diffusion model, but without fixing the start and goal positions (without the inpainting process)
        Args:
            n: number of paths to generate
            num_inference_steps: number of diffusion steps used during inference
            steer_length: must be set to True to use length steering during inference
            steer_obstacles: must be set to True to use obstacle steering during inference
            img_output_dir: output directory where svg images of the diffusion process will be stored

        Returns:
            result paths as numpy array of shape (n, path_length, 2)
        """
        seed_generator = torch.manual_seed(self.config.seed)
        noise_shape = (n, self.config.in_channels, self.config.sequence_length)
        noise_input = randn_tensor(noise_shape, generator=seed_generator, device=self.model.device)
        diffusion_result = evaluate(self.model, self.noise_scheduler, noise_input, seed_generator, self.model.device,
                                    self.env, num_inference_steps, None, steer_length, steer_obstacles,
                                    img_output_dir)
        paths = diffusion_result.cpu().detach().numpy()

        return np.transpose(paths, (0, 2, 1))  # reshaping paths from (n, 2, length) to (n, length, 2)
