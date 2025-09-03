import torch

from .data.collate import collate_x_dict
from .config import read_config
from .load import load_model_from_cfg
from hydra.utils import instantiate


def load_tmr_model_easy(device="cpu"):
    run_dir = "tmr_models/tmr_humanml3d_guoh3dfeats"
    ckpt_name = "last"
    cfg = read_config(run_dir)

    print("Loading the model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)
    normalizer = instantiate(cfg.data.motion_loader.normalizer)

    # print("Loading the text model")
    # text_model = instantiate(cfg.data.text_to_token_emb, preload=False, device=device)

    def easy_forward(motions_or_texts):
        if isinstance(motions_or_texts[0], str):
            raise NotImplementedError
            texts = motions_or_texts
            x_dict = collate_x_dict(text_model(texts))
        else:
            motions = motions_or_texts
            motions = [
                normalizer(torch.from_numpy(motion).to(torch.float)).to(device)
                for motion in motions
            ]
            x_dict = collate_x_dict(
                [
                    {
                        "x": motion,
                        "length": len(motion),
                    }
                    for motion in motions
                ]
            )

        with torch.inference_mode():
            latents = model.encode(x_dict, sample_mean=True).cpu()
        return latents

    return easy_forward
