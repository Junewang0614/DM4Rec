
from models import MultiLayerPNet
import gaussian_diffusion as gd
from respace import SpacedDiffusion,space_timesteps
def create_mlp_model(
        embedding_size=64,
        hidden_sizes = [64,128,256,128], # 中间层的大小
        dropout=0.0,
        class_cond = False,
        cond_embed = -1,
        learn_sigma = False
):
    input_size = embedding_size
    output_size = embedding_size if not learn_sigma else 2*embedding_size

    model = MultiLayerPNet(input_size,output_size,
                           hidden_sizes=hidden_sizes,
                           class_cond=class_cond,
                           cond_embed=cond_embed,
                           )
    return model

# create_diffusion
def create_gaussian_diffusion(
    *,
    diffusion_steps=64,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    steps = diffusion_steps
    betas = gd.get_named_beta_schedule(noise_schedule, steps)

    # 设置loss
    # 1. 是否使用kl散度
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    # 2. 是否要学习sigma，即方差
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    # 3. 默认状态下就是原始ddpm的，mse
    else:
        loss_type = gd.LossType.MSE

    # 对timestep要做的改进 TODO??
    if not timestep_respacing:
        timestep_respacing = [steps]

    # TODO??
    use_timesteps = space_timesteps(steps, timestep_respacing)

    return SpacedDiffusion(
        use_timesteps=use_timesteps,
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ), # 模型预测的target是什么
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),# 模型是否预测方差
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )