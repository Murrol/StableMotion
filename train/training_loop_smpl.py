import functools
import os
import blobfile as bf
from diffusion import logger
from utils import dist_util
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch import GradScaler, autocast
from ema_pytorch import EMA
from diffusion.resample import create_named_schedule_sampler


class TrainLoop:
    """
    Lightweight training loop for diffusion models with:
      - mixed precision (GradScaler + autocast),
      - AdamW + gradient clipping,
      - EMA (optional),
      - checkpoint save/resume,
      - simple inpainting mask manager,
      - per-timestep loss logging with quantiles.

    __init__ Args:
        args: Namespace of run configs (batch_size, lr, log/save intervals, etc.).
        train_platform: Object with `report_scalar(name, value, iteration, group_name)` for metrics.
        model: Torch model to train.
        diffusion: Diffusion wrapper providing `training_losses` and `num_timesteps`.
        data: Iterable of batches (each batch is a dict with at least 'x' and 'mask').

    Notes:
        - `weighted_loss`: if enabled in `args`, loads per-feature weights from `normalizer_dir/feature_w_file`.
        - EMA is enabled via `args.model_ema` and configured by related `*_ema_*` args.
        - Learning rate annealing stops training when `step + resume_step >= lr_anneal_steps`.
    """
    def __init__(self, args, train_platform, model, diffusion, data):
        self.args = args
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps
        self.gradient_clip = args.gradient_clip
        self.snr_gamma = args.snr_gamma
        self.use_l1 = args.l1_loss

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite
       
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.feature_w = None
        if args.weighted_loss:
            self.feature_w = torch.load(os.path.join(args.normalizer_dir, args.feature_w_file), weights_only=True)
            self.feature_w.requires_grad_(False)

        self._load_and_sync_parameters()

        self.scaler = GradScaler()

        self.opt = AdamW(
            list(self.model.parameters()), lr=self.lr, weight_decay=self.weight_decay
        )

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)

        self.ema_model = None
        if args.model_ema:
            self.ema_model = EMA(self.model,
                                 beta=args.model_ema_decay,
                                 update_every=args.model_ema_steps,
                                 update_after_step=args.model_ema_update_after,
                                 include_online_model=False,
                                 )
            if self.resume_step:
                self._load_ema_state()
        
    def _load_and_sync_parameters(self):
        # Load model weights from checkpoint if provided/found.
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                ),
                strict=False,
            )

    def _load_optimizer_state(self):
        # Load optimizer state paired with the resumed model step.
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _load_ema_state(self):
        # Load EMA weights paired with the resumed model step.
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"ema{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading ema state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.ema_model.load_state_dict(state_dict)

    def run_loop(self):
        # Main training loop with periodic logging/saving/eval.
        for epoch in range(self.num_epochs):
            for batch in tqdm(self.data, disable=True):
                batch = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in batch.items()}

                self.run_step(batch)
                if self.step % self.log_interval == 0:
                    for k,v in logger.get_current().name2val.items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.step+self.resume_step, group_name='Loss')

                if self.step % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    self.evaluate()
                    self.model.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1

                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break
                if self.step + self.resume_step >= self.num_steps:
                    break

            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
            if self.step + self.resume_step >= self.num_steps:
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    def evaluate(self):
        # Optional: evaluation during training.
        if not self.args.eval_during_training:
            return
        else:
            raise NotImplementedError

    # @torch.compile
    def run_step(self, batch):
        # One optimization step: forward, loss, backward, optimizer/EMA/step logs.
        self.forward_backward(batch)

        if self.gradient_clip:
            self.scaler.unscale_(self.opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # scaler.step() applies optimizer step if grads are finite; otherwise skips.
        self.scaler.step(self.opt)
        # Update scaler for next iteration.
        self.scaler.update()

        if self.ema_model is not None:
            self.ema_model.update()

        self.log_step()
    
    def mask_manager(self, batch, sample, sample_cond):
        """
        Build inpainting masks and attention masks.

        Modes (split by batch fraction):
          - detection mode (first part): all features except the last are conditioned (masked=0), last is to generate.
          - inpaint mode (second part): last feature is conditioned (masked=0), others randomly masked per sequence.
        """
        B, D, N = sample.shape

        bs_1interval = B // self.args.fraction
        bs_2interval = B - bs_1interval

        inpaint_cond = torch.ones_like(sample) # true to generate, false as condition

        ### det mode ###
        inpaint_cond[:bs_1interval, :-1] = 0

        ### inpaint mode ###
        inpaint_cond[-bs_2interval: , -1] = 0

        ntokens = N
        rand_time = uniform((bs_2interval,), device=self.device)
        rand_mask_probs = cosine_schedule(rand_time)
        num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=5) 
        # nan BUG if sum(mask) == 0 after inpaint_cond = inpaint_cond.bool() & sample_cond['attention_mask'].unsqueeze(-2)

        batch_randperm = torch.rand((bs_2interval, ntokens), device=self.device).argsort(dim=-1)
        
        unmask = batch_randperm > num_token_masked.unsqueeze(-1)
        unmask = unmask.unsqueeze(-2).repeat(1, D, 1)

        inpaint_cond[-bs_2interval:] = torch.where(unmask, 0, inpaint_cond[-bs_2interval:])
        
        # attention mask in self attention to ignore padding tokens
        sample_cond['attention_mask'] = batch['mask'].squeeze().bool().clone() # -> bs, seqlen

        # indicate which feature are observed as condition (False), which are to be inpainted (True)
        # also used to specify the working mode, i.e. detection mode or inpainting mode
        inpaint_cond = inpaint_cond.bool() & sample_cond['attention_mask'].unsqueeze(-2)
        sample_cond['inpaint_cond'] = inpaint_cond
        
        # used in gaussion diffusion training loss compute
        sample_cond['y'] = {'mask': inpaint_cond} # bs, C, seqlen 


    def forward_backward(self, batch): #Training basic step
        self.opt.zero_grad()

        # Forward under autocast; compute diffusion loss; scaled backward.
        with autocast(device_type='cuda', dtype=torch.float16):
            for i in range(0, batch['x'].shape[0], self.microbatch):
                
                # Eliminates the microbatch feature
                assert i == 0
                assert self.microbatch == self.batch_size
                sample = batch['x']
                sample_cond = {}
  
                B, D, N = sample.shape
                
                # Prepare feature weighting (if enabled)
                # TODO Fix the ugly hardcode here
                feature_w = self.feature_w.to(dist_util.dev())[None,:,None].repeat(B, 1, N) if self.feature_w is not None else None #[dim] -> [bs, dim, l]

                t = torch.randint(low=0, high=self.diffusion.num_timesteps, size=(sample.shape[0],), device=dist_util.dev())

                self.mask_manager(batch, sample, sample_cond)
                
                # from PIL import Image # Debug plotting
                # Image.fromarray(sample_cond['y']['mask'][:, 0].detach().cpu().numpy()).save('temp_inpaintmask.bmp')
                # Image.fromarray(sample_cond['y']['mask'][:, -1].detach().cpu().numpy()).save('temp_inpaintmasklast.bmp')
                # Image.fromarray(sample_cond['attention_mask'].detach().cpu().numpy()).save('temp_attmask.bmp')
                # raise
                

                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.model,
                    sample,  # [bs, ch, image_size, image_size]
                    t,  # [bs] sampled timesteps
                    model_kwargs=sample_cond,
                    feature_w=feature_w,
                    snr_gamma=self.snr_gamma,
                    use_l1=self.use_l1,
                )

                losses = compute_losses()
                
                loss = (losses["loss"]).mean()
                log_loss_dict(
                    self.diffusion, t, {k: v for k, v in losses.items()}
                )

                # Scaled backward for AMP.
                self.scaler.scale(loss).backward()

    def log_step(self):
        # Lightweight counters for external loggers.
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


    def ckpt_file_name(self):
        # Step-indexed checkpoint filename.
        return f"model{(self.step+self.resume_step):09d}.pt"


    def save(self):
        # Save model/optimizer/EMA (if any) to blob storage/local path.
        # def save_checkpoint(params):
        #     state_dict = self.mp_trainer.master_params_to_state_dict(params)
        def save_checkpoint(model):
            state_dict = model.state_dict()

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        # save_checkpoint(self.mp_trainer.master_params)
        save_checkpoint(self.model)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)
        if self.ema_model is not None:
            with bf.BlobFile(
                bf.join(self.save_dir, f"ema{(self.step+self.resume_step):09d}.pt"),
                "wb",
            ) as f:
                torch.save(self.ema_model.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse '.../modelNNNNNN.pt' and return the integer step. Returns 0 on failure.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # Override to redirect checkpoints to external storage if needed.
    return logger.get_dir()


def find_resume_checkpoint():
    # Hook for infra-specific checkpoint discovery.
    return None


def log_loss_dict(diffusion, ts, losses):
    # Log mean loss and per-quartile buckets over timesteps.
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # quartiles over the diffusion time index
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

def uniform(shape, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)

import math
def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)