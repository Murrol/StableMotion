from model.stablemotion import StableMotionDiTModel
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

def create_model_and_diffusion(args):
    model = StableMotionDiTModel( 
        in_channels=233,
        out_channels=233,
        num_layers=args.layers, # default 8
        num_attention_heads=args.heads, # default 8
        attention_head_dim=64,
        class_cond=True, # working mode indicator as class
        zero_init=args.zero_init if hasattr(args, 'zero_init') else False,
    )
    
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion

def create_gaussian_diffusion(args):
    # default params
    predict_xstart = args.predict_xstart  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = args.ts_respace if hasattr(args, 'ts_respace') else '' #''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )