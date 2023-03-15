import os.path
import torch
import sys
import argparse
sys.path.append('../source')
sys.path.append('../')

# from source.script_util import seed_torch
# diffusion部分
from source import dist_util
from source import logger
from source.script_util import (
    diffusion_defaults,mlp_model_defaults,
    add_dict_to_argparser,args_to_dict,seed_torch,create_path,dict2str
)
from source.creation import create_mlp_model,create_gaussian_diffusion
from source.resample import create_named_schedule_sampler
from source.trainer_v2 import Trainer_v2
# gru部分
import logging
from logging import getLogger
from recbole.config import Config as ReConfig
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, init_logger,get_model
from recbole.model.sequential_recommender import GRU4Rec
import recbole.utils.logger as Reclogger
# from recbole.trainer import Trainer

def main():
    seed_torch()

    model_name = 'GRU4Rec'
    dataset_name = 'ml-100k'

    # NOTE: 1.diffusion的初始化
    args, model_keys, diffusion_keys = create_argparser()
    args = args.parse_args()
    dist_util.setup_dist()
    model_keys = list(model_keys) + ['learn_sigma']  # 这个用来判断输出大小

    # NOTE: 扩散中的神经网络
    model = create_mlp_model(**args_to_dict(args, model_keys))
    # NOTE: 扩散框架
    diffusion = create_gaussian_diffusion(**args_to_dict(args, diffusion_keys))
    # NOTE: 时间步采样器
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # NOTE: 2.gru的配置与数据集构造
    cond_config = create_recbole_config(model_name,dataset_name)
    init_seed(cond_config['seed'],cond_config['reproducibility'])
    dataset = create_dataset(cond_config)
    train_data,valid_data,test_data = data_preparation(cond_config,dataset)

    # NOTE: 3.condition网络
    gru_model = GRU4Rec(cond_config,train_data.dataset).to(cond_config['device'])

    # TODO: 4.TRAINER
    trainer = Trainer_v2(
        dataset=dataset,device=dist_util.dev(),
        noise_model=model,diffusion_model=diffusion,schedule_sampler=schedule_sampler,
        condition_model=gru_model,diff_config=args,rec_config=cond_config
    )

    # TODO: 5.LOGGER
    init_logger(cond_config)
    reclogger = getLogger()
    # 本身自带一个sh
    # c_handler = logging.StreamHandler()
    # c_handler.setLevel(logging.DEBUG)
    # reclogger.addHandler(c_handler)

    # loss文件
    create_path(os.path.join(trainer.saved_model_path,'log'))
    log_file = os.path.join(trainer.saved_model_path,'log','output.log') # 日志文件路径
    f_handler = logging.FileHandler(log_file)
    f_handler.setLevel(logging.INFO)
    reclogger.addHandler(f_handler)

    trainer.logger = reclogger

    # 控制台输出
    reclogger.debug(cond_config) # 输出cond_config
    reclogger.debug('============The condition network============')
    reclogger.debug(gru_model)
    reclogger.debug('============The diffusion network============')
    reclogger.debug(model)

    # NOTE:6. Training
    best_valid_results = trainer.train_loop_final(train_data=train_data,valid_data=valid_data,show_progress=False)
    # best_valid_results = trainer.train_loop_diff(train_data=train_data,valid_data=valid_data,show_progress=False)
    # best_valid_results = trainer.train_loop_gru(train_data=train_data,valid_data=valid_data,show_progress=False)

    # 最后结果存一下
    valid_result_output = (Reclogger.set_color('best results in valid data: \n', 'blue'))
    valid_result_output += dict2str(best_valid_results)
    valid_gru_result_output = (Reclogger.set_color('best results in valid data with condition model: \n', 'blue'))
    valid_gru_result_output += dict2str(trainer.best_cond_valid_results)
    reclogger.info('================The results====================')
    reclogger.info(valid_result_output)
    reclogger.info(valid_gru_result_output)

    # TODO: 7. Testing
    # 和检验的步骤一样
    # 模型重新加载一下
    cond_checkpoint = torch.load(os.path.join(trainer.saved_model_path,'condition_model.pt'))
    diff_checkpoint = torch.load(os.path.join(trainer.saved_model_path,'noise_model.pt'))
    trainer.condition_model.load_state_dict(cond_checkpoint['state_dict'])
    reclogger.info('Reload the condition model successfully.')
    trainer.noise_model.load_state_dict(diff_checkpoint)
    reclogger.info('Reload the noise model successfully.')

    reclogger.info('================ The test results ===============')
    test_results,test_score = trainer.diff_eval_epoch(test_data,show_progress=True)
    test_result_output = (Reclogger.set_color('The results in test data: \n', 'blue'))
    test_result_output += dict2str(test_results)
    reclogger.info(test_result_output)
    print(test_results)

# NOTE: gru和数据集的配置
def create_recbole_config(model,dataset,filepath=None):
    # TODO:也可以从外部导入配置文件
    parameter_dict = {
        'data_path': "/data03/wangyidan-slurm/RecBole/dataset/", # 目录下有对应数据集名称的文件夹，里面有.inter原子文件
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'user_inter_num_interval': "[3,inf)",  #
        'item_inter_num_interval': "[3,inf)",  #
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
        'neg_sampling': None,
        'epochs': 200,
        'train_batch_size': 8192,
        'stopping_step':50,
        'topk': [5, 10, 20],
        'metrics': ["Recall", "MRR", "NDCG", "Precision"],
        'learning_rate': 0.001,
        'loss_type': 'BPR',
        'learner':'adam',
        'state':'debug',
        'checkpoint_dir':'../saved',
        #
        'eval_args': {
            'split': {'LS': 'valid_and_test'},
            'group_by': 'user',
            'order': 'TO',
            'mode': 'full'}
    }
    config = ReConfig(model=model, dataset=dataset, config_dict=parameter_dict)

    return config

# NOTE: diffusion的配置
def create_argparser():
    '''从超参数字典中自动生成命令行传参的 argument parser'''
    # NOTE:这里是和训练相关的超参数

    defaults = dict(
        clip_denoised=True, # 采样用
        schedule_sampler="uniform", # diffusion的时间步采样器
        data_dir = "", # 训练数据目录
        data_set = "", # 数据集名称，后面可以用于调用分开处理的函数
        lr = 1e-3, #训练学习率
        weight_decay=0.0, # 衰减率
        lr_anneal_steps=0, # 暂时不清楚,和学习率的更新有关的
        # batch划分相关
        batch_size=1,
        microbatch=-1, # 训练时会将batch再划分
        # 迭代相关,都没用
        iterations=150000, #
        log_interval=10, # 记录loss用
        save_interval=10000, # 保存模型用
        # 不太懂
        ema_rate="0.9999",
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    # TODO:添加对模型类别的判断，从而调用不同的模型加载函数
    model_keys = mlp_model_defaults().keys()
    defaults.update(mlp_model_defaults())

    diffusion_keys = diffusion_defaults().keys()
    defaults.update(diffusion_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser,defaults)

    return parser,model_keys,diffusion_keys

if __name__ == '__main__':
    main()