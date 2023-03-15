import argparse
import torch
import os
from copy import deepcopy
import operator
import datetime
import random
import numpy as np

def mlp_model_defaults():
    '''神经网络的超参数配置'''
    # 先写mlp
    # mlp 设置成输入每层的嵌入数，然后自动生成网络的形式
    # 注意，因为对于diffusion来说，输入输出大小一样，所以只用一个embedding_size
    res = dict(
        embedding_size=64,
        # hidden_sizes = [64,128,256,128], # 中间层的大小
        hidden_sizes=[64, 128],
        dropout=0.0,
        class_cond = True, # 是否添加额外的条件
        cond_embed = 64,
    )
    return res

def diffusion_defaults():
    '''diffusion框架的参数配置'''

    res = dict(
        diffusion_steps = 1000, # 扩散步数
        noise_schedule = "linear", # beta生成方式
        # 是否学习sigma
        learn_sigma=False,
        rescale_learned_sigmas=False,
        # 学习目标是否为其他的
        predict_xstart=False,
        # loss function 相关
        use_kl=False,
        # TODO:时间步相关，但目前不太懂
        rescale_timesteps=False,
        timestep_respacing="",
    )

    return res

# ======================TOOL FUNCTIONS============================

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

# 将params转变为 state dict
def _master_params_to_state_dict(model, params):
    # if self.use_fp16:
    #     master_params = unflatten_master_params(
    #         self.model.parameters(), master_params
    #     )
    state_dict = model.state_dict()
    for i, (name, _value) in enumerate(model.named_parameters()):
        assert name in state_dict
        state_dict[name] = params[i]
    return state_dict

    # 保存模型
def save_checkpoint(path,
                    noise_model=None,
                    condition_model_dict=None):
    if noise_model is not None:
        # 1. 保存noise
        noise_model_file = os.path.join(path,'noise_model.pt')
        torch.save(noise_model.state_dict(), noise_model_file)
        print('{} is saved'.format(noise_model_file))
    if condition_model_dict is not None:
        # 2. 保存gru
        condition_model_file = os.path.join(path, 'condition_model.pt')
        # 保存字典
        condition_state = {
            "rec_config": condition_model_dict['config'],
            "state_dict": condition_model_dict['model'].state_dict(),
            # "best_valid_result": valid_result, todo
        }
        torch.save(condition_state, condition_model_file, pickle_protocol=4)

        print('{} is saved'.format(condition_model_file))

# TODO：修改成比较字典的
def early_stopping_results(value, best, cur_step, max_step, bigger=True):
    r"""validation-based early stopping,according to results

    Args:
        value (dict): current result
        best (dict): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if best is None:
        best = deepcopy(value)
        update_flag=True
        cur_step = 0
        # Todo 这里要验证一下，best是否能保持
        return best, cur_step, stop_flag, update_flag

    assert type(value) == type(best),'the cur valid results has different type with the best results.'

    # 两个dict这里进行比较:
    assert operator.eq(value.keys(),best.keys()),'the cur valid results {} has different keys with the best results{}.'.format(value.keys(),best.keys())
    bignum = 0
    bigflag = False
    for key in value.keys():
        if value[key] > best[key]:
            bignum += 1
    if 2*bignum > len(value.keys()):
        bigflag = True

    if bigger:
        if bigflag:
            cur_step = 0
            best = deepcopy(value)
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if not bigflag:
            cur_step = 0
            best = deepcopy(value)
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def dict2str(result_dict):
    r"""convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    return "    ".join(
        [str(metric) + " : " + str(value) for metric, value in result_dict.items()]
    )