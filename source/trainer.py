from tqdm import tqdm
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
import recbole.utils.logger as Reclogger
import recbole.utils.utils as Recutils
from recbole.utils import EvaluatorType,calculate_valid_score
from recbole.evaluator import Evaluator, Collector
from recbole.model.loss import BPRLoss
from torch.nn import CrossEntropyLoss
import numpy as np
import os
from source.script_util import early_stopping_results,save_checkpoint,get_local_time,create_path
from copy import deepcopy
class Trainer(object):
    def __init__(self,
                dataset,
                noise_model,
                diffusion_model,
                device,
                diff_config, # diffusion框架下的config，使用点来访问
                rec_config,
                # loss_type='', # 这个由diffusion决定
                condition_model=None,
                schedule_sampler=None,):

        self.dataset = dataset
        self.noise_model = noise_model
        self.diffusion_model = diffusion_model
        self.condition_model = condition_model
        self.schedule_sampler = schedule_sampler # 照搬iddpm就可以
        self.device = device
        self.diff_config = diff_config
        self.rec_config = rec_config

        #TODO: 初始化优化器
        #优化器的4个参数,这里先不选类型了，就是adamW
        self.model_params = list(self.noise_model.parameters()) + list(self.condition_model.parameters())
        # 是根据diff来选择的
        self.learning_rate = diff_config.lr
        self.weight_decay = diff_config.weight_decay

        self.optimizer = AdamW(self.model_params,lr = self.learning_rate,weight_decay=self.weight_decay)

        self.lr_anneal_steps = diff_config.lr_anneal_steps


        # TODO: 一些训练设置
        self.epoch_size = rec_config['epochs']
        self.cur_epoch = 0
        self.enable_scaler = torch.cuda.is_available() and self.rec_config["enable_scaler"] # 加速设置
        self.train_loss_dict = {}

        # TODO: 评测检验使用，都是基于recbole的
        self.eval_collector = Collector(rec_config)  #
        self.evaluator = Evaluator(rec_config)  #
        self.eval_step = min(rec_config["eval_step"], self.epoch_size)
        self.stopping_step = rec_config["stopping_step"]

        # TODO: 保存模型的设置
        # 这里有两个最好是最好的模型结构
        self.saved_model_path = os.path.join(rec_config["checkpoint_dir"],get_local_time()) # 模型保存的路径，一定是全称
        create_path(self.saved_model_path)
        self.best_valid_results = None
        self.best_cond_valid_results = None

        # 模型进入相应的设备
        self.ddp_noise_model = DDP(
                self.noise_model.to(self.device),
                device_ids=[self.device],
                output_device=self.device,
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        self.ddp_condition_model = DDP(
            self.condition_model,
            device_ids=[self.device],
            output_device=self.device,
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=False,
        )


    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def train_loop(self,
                   train_data,
                   valid_data,
                   show_progress=True):
        for epoch_idx in range(0,self.epoch_size):
            # 1. 训练一个epoch
            train_loss = self.train_epoch(train_data,epoch_idx,show_progress=show_progress)
            # self.train_loss_dict[epoch_idx] = (
            #     sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            # )
            self.train_loss_dict[epoch_idx] = train_loss # 本身是dict

            # 2. 检验一个epoch
            # TODO:当没有valid_data的时候依据loss的结果来更新模型
            valid_result,valid_score = self.eval_epoch(valid_data,show_progress=show_progress)
            # early stopping的设置
            (
                self.best_valid_results,
                self.cur_epoch,
                stop_flag,
                update_flag,
            ) = early_stopping_results(
                valid_result,
                self.best_valid_results,
                self.cur_epoch,
                max_step=self.stopping_step,
                bigger=self.rec_config["valid_metric_bigger"],
            )

            # 3. 更新保存模型，停止训练
            if update_flag:
                cond_dict = dict(
                    config = self.rec_config,
                    model = self.condition_model
                )
                print('now is saving NO {} EPOCH'.format(epoch_idx))
                save_checkpoint(self.saved_model_path, noise_model=self.noise_model, condition_model_dict=cond_dict)
                self.best_valid_results = deepcopy(valid_result)

            if stop_flag:
                # stop_output = "Finished training, best eval result in epoch %d" % (
                #         epoch_idx - self.cur_step * self.eval_step
                # )
                # if verbose:
                #     self.logger.info(stop_output)
                print('stopping epoch idx is {}'.format(epoch_idx))
                break

        return self.best_valid_results

    # 一定是训练数据
    # 训练一个epoch的函数
    def train_epoch(self,
                 data,
                 epoch_idx,
                 loss_func = None,
                 show_progress=True):
        # data格式：字典，里面包括多列数据
        self.noise_model.train()
        self.condition_model.train()
        iter_data = (
            tqdm(
                data,
                total=len(data),
                ncols=100,
                desc=f"Train {epoch_idx}",
            )
            if show_progress
            else data
        )

        total_losses = None
        scaler = GradScaler(enabled=self.enable_scaler)
        for batch_idx,batch in enumerate(iter_data):
            batch = batch.to(self.device)
            # self.optimizer.zero_grad()
            self.zero_grad() # 优化器更新

            # 先获得gru的输出
            item_seq = batch['item_id_list'] # 序列
            item_len = batch['item_length'] # 长度
            cond_output = self.ddp_condition_model(item_seq,item_len) # [batch_size,embedding] # gru输出
            target_item = batch['item_id'].to(self.device) # [batch_size]

            target_item_embed = self.condition_model.item_embedding(target_item)
            # diffusion 框架进行运算
            cond = {
                'y':cond_output.to(self.device) 
            }
            t,weights = self.schedule_sampler.sample(cond_output.shape[0],self.device) # batch_size[]

            # TODO: loss计算
            diffusion_losses = self.diffusion_model.training_losses(
                self.ddp_noise_model,
                target_item_embed,
                t,
                model_kwargs=cond) # 这是个字典
            diffusion_loss = (diffusion_losses["loss"] * weights).mean()
            # TODO:加gru的loss，先是bpr
            neg_items = batch['neg_item_id']
            pos_items_embed = target_item_embed # [batch,embed_size]
            neg_items_embed = self.condition_model.item_embedding(neg_items) # [batch,embed_size]
            cond_loss = self.bpr_loss(cond_output,pos_items_embed,neg_items_embed)
            # 总loss
            total_loss = diffusion_loss + cond_loss
            self._check_nan(diffusion_loss)
            self._check_nan(cond_loss)
            # diffusion_loss.backward()
            scaler.scale(total_loss).backward()

            total_losses = {
                'diffusion_loss':diffusion_loss.item() if total_losses is None else total_losses['diffusion_loss'] + diffusion_loss.item(),
                'condition_loss':cond_loss.item() if total_losses is None else total_losses['condition_loss'] + cond_loss.item(),
                'total_loss':total_loss.item() if total_losses is None else total_losses['total_loss'] + total_loss.item(),
            }

            # 优化器优化,step等
            # TODO: 优化器如果要更新lr的话，可以参考iddpm里面的self._anneal_lr()
            scaler.step(self.optimizer) # 优化器优化
            scaler.update()

            # gpu使用的输出
            if self.device.type == 'cuda' and show_progress:

                iter_data.set_postfix_str(
                    Reclogger.set_color("GPU RAM: " + Recutils.get_gpu_usage(self.device), "yellow")
                )

        return total_losses

    @torch.no_grad()
    def eval_epoch(self,
                   valid_data=None,
                   show_progress=True):
        if not valid_data:
            return

        self.noise_model.eval()
        self.condition_model.eval()

        # TODO:要确定loss function

        # 要统计所有的item的数量
        if self.rec_config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = valid_data._dataset.item_num

        iter_data = (
            tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=Reclogger.set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else valid_data
        )

        # item的嵌入向量
        test_items_emb = self.condition_model.item_embedding.weight
        # 这里要除以item自己的向量长度 [item_size]的大小
        # 平方以后求和，开方,这里可以提前放
        item_all_emb = torch.mul(test_items_emb, test_items_emb).sum(dim=-1) # [item_num]
        item_all_emb = torch.sqrt(item_all_emb) # [item_num]


        for batch_idx,batch in enumerate(iter_data):
            # 这个batch 更复杂一些，本身包括data的list，还包括了target
            # interaction是包括了data的dict
            interaction, history_index, positive_u, positive_i = batch

            # NOTE:1. 构造condition输入
            item_seq = interaction['item_id_list']
            item_len = interaction['item_length']
            cond_output = self.ddp_condition_model(item_seq, item_len)
            cond = {
                'y': cond_output.to(self.device)
            }

            # NOTE:2. 经过diffusion的框架
            # 采样shape [batch_size,embedding_size]
            sample_shape = (cond_output.shape[0],self.rec_config['embedding_size'])
            sample = self.diffusion_model.p_sample_loop(
                self.noise_model,
                sample_shape,
                clip_denoised=self.diff_config.clip_denoised,
                model_kwargs=cond
            ) # 这个的大小？ 对应的目标 item [batch_size,item_embedding]

            # print(sample.shape)

            # NOTE:3 找top k
            # 相似度得分
            scores = torch.matmul(
                sample, test_items_emb.transpose(0, 1)
            )
            scores = torch.div(scores,item_all_emb) # 余弦相似度
            scores = scores.view(-1, self.tot_item_num) # 一些处理
            scores[:, 0] = -np.inf
            # print(scores.shape)

            if self.device.type == 'cuda' and show_progress:
                iter_data.set_postfix_str(
                    Reclogger.set_color("GPU RAM: " + Recutils.get_gpu_usage(self.device), "yellow")
                )

            self.eval_collector.eval_batch_collect(
                scores, interaction, positive_u, positive_i
            )

        struct = self.eval_collector.get_data_struct()
        valid_result = self.evaluator.evaluate(struct)
        valid_score = calculate_valid_score(valid_result, self.rec_config["valid_metric"].lower())
        return valid_result,valid_score

    # def save_checkpoint(self,valid_result):
    #     # 保存两个config
    #     state = {
    #         "rec_config": self.rec_config,
    #         "diff_config": self.diff_config,
    #         "best_valid_result":valid_result,
    #         "state_dict": self.model.state_dict(),
    #         "other_parameter": self.model.other_parameter(),
    #         "optimizer": self.optimizer.state_dict(),
    #     }
    #     torch.save(state, saved_model_file, pickle_protocol=4)



    def zero_grad(self):
        for param in self.model_params:
            # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
    

    def bpr_loss(self,output,pos_items_embed,neg_items_embed):
        '''size:[batch_size,embed_size]'''
        loss_fun = BPRLoss()
        pos_score = torch.sum(output * pos_items_embed, dim=-1)
        neg_score = torch.sum(output * neg_items_embed, dim=-1)
        loss = loss_fun(pos_score, neg_score)

        return loss

    # TODO: CE LOSS
    # def ce_loss
    def ce_loss(self,output,target_items):
        loss_fun = CrossEntropyLoss()
        item_emb_weight = self.condition_model.item_embedding.weight
        logits = torch.matmul(output,item_emb_weight.transpose(0,1))
        loss = loss_fun(logits,target_items)

        return loss


# Trainer for check

class Check_Cond_Trainer(Trainer):
    def __init__(self,
                 dataset,
                 noise_model,
                 diffusion_model,
                 device,
                 diff_config,  # diffusion框架下的config，使用点来访问
                 rec_config,
                 # loss_type='', # 这个由diffusion决定
                 condition_model=None,
                 schedule_sampler=None, ):
        super(Check_Cond_Trainer, self).__init__(dataset,noise_model,diffusion_model,device,diff_config,rec_config,condition_model,schedule_sampler)

    # 重写几个函数

    # train中主要是将total_loss = cond loss
    # 不考虑diffusion loss
    def train_epoch(self,
                 data,
                 epoch_idx,
                 loss_func = None,
                 show_progress=True):
        # data格式：字典，里面包括多列数据
        self.noise_model.train()
        self.condition_model.train()
        iter_data = (
            tqdm(
                data,
                total=len(data),
                ncols=100,
                desc=f"Train {epoch_idx}",
            )
            if show_progress
            else data
        )

        total_losses = None
        scaler = GradScaler(enabled=self.enable_scaler)
        for batch_idx,batch in enumerate(iter_data):
            batch = batch.to(self.device)
            # self.optimizer.zero_grad()
            self.zero_grad() # 优化器更新

            # 先获得gru的输出
            item_seq = batch['item_id_list'] # 序列
            item_len = batch['item_length'] # 长度
            cond_output = self.ddp_condition_model(item_seq,item_len) # [batch_size,embedding] # gru输出
            target_item = batch['item_id'].to(self.device) # [batch_size]

            target_item_embed = self.condition_model.item_embedding(target_item)
            # diffusion 框架进行运算
            cond = {
                'y':cond_output.to(self.device)
            }
            t,weights = self.schedule_sampler.sample(cond_output.shape[0],self.device) # batch_size[]

            # NOTE: 这里total loss 就是condition loss。和diffusion无关
            diffusion_losses = self.diffusion_model.training_losses(
                self.ddp_noise_model,
                target_item_embed,
                t,
                model_kwargs=cond) # 这是个字典
            diffusion_loss = (diffusion_losses["loss"] * weights).mean()

            neg_items = batch['neg_item_id']
            pos_items_embed = target_item_embed # [batch,embed_size]
            neg_items_embed = self.condition_model.item_embedding(neg_items) # [batch,embed_size]
            cond_loss = self.bpr_loss(cond_output,pos_items_embed,neg_items_embed)
            # NOTE: 总loss condition loss
            total_loss = cond_loss
            self._check_nan(diffusion_loss)
            self._check_nan(cond_loss)
            # diffusion_loss.backward()
            scaler.scale(total_loss).backward()

            total_losses = {
                'diffusion_loss':diffusion_loss.item() if total_losses is None else total_losses['diffusion_loss'] + diffusion_loss.item(),
                'condition_loss':cond_loss.item() if total_losses is None else total_losses['condition_loss'] + cond_loss.item(),
                'total_loss':total_loss.item() if total_losses is None else total_losses['total_loss'] + total_loss.item(),
            }

            # 优化器优化,step等
            # TODO: 优化器如果要更新lr的话，可以参考iddpm里面的self._anneal_lr()
            scaler.step(self.optimizer) # 优化器优化
            scaler.update()

            # gpu使用的输出
            if self.device.type == 'cuda' and show_progress:
                iter_data.set_postfix_str(
                    Reclogger.set_color("GPU RAM: " + Recutils.get_gpu_usage(self.device), "yellow")
                )

        return total_losses


    # eval中直接使用gru的网络的结果作为推理用的分数
    @torch.no_grad()
    def eval_epoch(self,
                   valid_data=None,
                   show_progress=True):
        if not valid_data:
            return

        self.noise_model.eval()
        self.condition_model.eval()

        # TODO:要确定loss function

        # 要统计所有的item的数量
        if self.rec_config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = valid_data._dataset.item_num

        iter_data = (
            tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=Reclogger.set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else valid_data
        )

        # # item的嵌入向量
        # test_items_emb = self.condition_model.item_embedding.weight
        # # 这里要除以item自己的向量长度 [item_size]的大小
        # # 平方以后求和，开方,这里可以提前放
        # item_all_emb = torch.mul(test_items_emb, test_items_emb).sum(dim=-1)  # [item_num]
        # item_all_emb = torch.sqrt(item_all_emb)  # [item_num]

        for batch_idx, batch in enumerate(iter_data):
            # 这个batch 更复杂一些，本身包括data的list，还包括了target
            # interaction是包括了data的dict
            interaction, history_index, positive_u, positive_i = batch

            # NOTE:1. 构造condition输入
            item_seq = interaction['item_id_list']
            item_len = interaction['item_length']
            cond_output = self.ddp_condition_model(item_seq, item_len) # gru的得分
            cond = {
                'y': cond_output.to(self.device)
            }

            # # NOTE:2. 经过diffusion的框架
            # # 采样shape [batch_size,embedding_size]
            # sample_shape = (cond_output.shape[0], self.rec_config['embedding_size'])
            # sample = self.diffusion_model.p_sample_loop(
            #     self.noise_model,
            #     sample_shape,
            #     clip_denoised=self.diff_config.clip_denoised,
            #     model_kwargs=cond
            # )  # 这个的大小？ 对应的目标 item [batch_size,item_embedding]

            # print(sample.shape)

            # # NOTE:3 找top k
            # # 相似度得分
            # scores = torch.matmul(
            #     sample, test_items_emb.transpose(0, 1)
            # )
            # scores = torch.div(scores, item_all_emb)  # 余弦相似度
            # scores = scores.view(-1, self.tot_item_num)  # 一些处理
            # scores[:, 0] = -np.inf

            # gru下的scores
            test_items_emb = self.condition_model.item_embedding.weight
            scores = torch.matmul(
                cond_output, test_items_emb.transpose(0, 1)
            )

            print(scores.shape) # [batch,item_size]

            if self.device.type == 'cuda' and show_progress:
                iter_data.set_postfix_str(
                    Reclogger.set_color("GPU RAM: " + Recutils.get_gpu_usage(self.device), "yellow")
                )

            self.eval_collector.eval_batch_collect(
                scores, interaction, positive_u, positive_i
            )

        struct = self.eval_collector.get_data_struct()
        valid_result = self.evaluator.evaluate(struct)
        valid_score = calculate_valid_score(valid_result, self.rec_config["valid_metric"].lower())
        return valid_result, valid_score









