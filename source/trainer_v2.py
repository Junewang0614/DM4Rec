from tqdm import tqdm
from tqdm.auto import trange
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
import recbole.utils.logger as Reclogger
import recbole.utils.utils as Recutils
from recbole.utils import EvaluatorType,calculate_valid_score
from recbole.evaluator import Evaluator, Collector
from recbole.model.loss import BPRLoss
import numpy as np
import os
from source.script_util import early_stopping_results,save_checkpoint,get_local_time,create_path
from copy import deepcopy

from source.trainer import Trainer

class Trainer_v2(Trainer):
    def __init__(self,
                 dataset,
                 noise_model,
                 diffusion_model,
                 device,
                 diff_config,  # diffusion框架下的config，使用点来访问
                 rec_config,
                 # loss_type='', # 这个由diffusion决定
                 condition_model=None,
                 schedule_sampler=None,
                 ):
        super(Trainer_v2, self).__init__(dataset=dataset,
                                         noise_model=noise_model,
                                         diffusion_model=diffusion_model,
                                         device=device,
                                         diff_config=diff_config,
                                         rec_config=rec_config,
                                         condition_model=condition_model,
                                         schedule_sampler=schedule_sampler)

        self.optimizer = AdamW(
            [
                {'params':self.noise_model.parameters()},
                {'params':self.condition_model.parameters(),'cond':True},
            ],
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.logger = None

        # 先不分布式了
        self.ddp_condition_model = self.condition_model

    # TODO: 重写training函数
    def diff_train_epoch(self,data,
                         epoch_idx,
                         loss_func = 'BPR',
                         show_progress = True):
        # diffusion中的神经网络是训练的
        # condition中的网络是固定的
        self.noise_model.train()
        self.condition_model.train()

        # 训练的数据
        iter_data = (
            tqdm(
                data,
                total=len(data),
                ncols=100,
                desc=Reclogger.set_color(f"Train {epoch_idx}", "pink"),
                # leave=False,
            )
            if show_progress
            else data
        )

        total_losses = None
        scaler = GradScaler(enabled=self.enable_scaler)

        for batch_idx,batch in enumerate(iter_data):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # NOTE: 构造condition
            item_seq = batch['item_id_list'] # seq
            item_len = batch['item_length'] # len
            cond_output = self.ddp_condition_model(item_seq,item_len)

            # TODO:调整gru输出梯度的运算
            cond_copy = cond_output
            # cond_copy = cond_output.detach()
            cond = {
                'y':cond_copy.to(self.device)
            }

            # NOTE:diffusion的运算
            target_item = batch['item_id'].to(self.device)
            target_item_embed = self.condition_model.item_embedding(target_item)
            t, weights = self.schedule_sampler.sample(cond_output.shape[0], self.device)

            # 输入，神经网络，target_item，t，cond
            # 获得noise，获得x_0
            # 'noise','xstart'
            predict_dict = self.get_eps_xstart_from_forward(target_item_embed,
                                                            t=t,
                                                            model_kwargs=cond)
            # NOTE: loss的计算
            assert cond_copy.shape == predict_dict['xstart'].shape,'condition output has different shape{} with xstart\'s{}.'.format(cond_output.shape,predict_dict['xstart'].shape)
            predict_item = cond_copy + predict_dict['xstart']

            # loss
            if loss_func == 'BPR':
                neg_items = batch['neg_item_id']
                neg_items_emb = self.condition_model.item_embedding(neg_items)
                pos_items_emb = self.condition_model.item_embedding(target_item)

                loss = self.bpr_loss(predict_item, pos_items_embed=pos_items_emb, neg_items_embed=neg_items_emb)
            elif loss_func == 'CE':
                loss = self.ce_loss(predict_item, target_items=target_item)
            else:
                raise NotImplementedError("Make sure 'loss_func' in ['BPR', 'CE']!")

            total_loss = loss

            # NOTE: 反向传播
            self._check_nan(total_loss)
            scaler.scale(total_loss).backward()
            total_losses = total_loss.item() if total_losses is None else total_losses + total_loss.item()

            # NOTE: 优化
            scaler.step(self.optimizer)
            scaler.update()

            # 添加loss在进度条的输出
            # if show_progress:
            #     iter_data.set_postfix_str(
            #         Reclogger.set_color(f"LOSS: {total_losses}", "yellow")
            #     )

            if self.device.type == 'cuda' and show_progress:
                # 改成后缀
                output_dict = {'LOSS':total_losses,"GRU RAM":Recutils.get_gpu_usage(self.device)}
                iter_data.set_postfix(output_dict)

                # iter_data.set_postfix_str(
                #     Reclogger.set_color("LOSS: %.4f GPU RAM" + Recutils.get_gpu_usage(self.device), "yellow") % (
                #         total_losses)
                # )

        return total_losses


    # TODO: 重写eval函数
    # 和 gru的eval主要的区别是推理过程添加了diffusion的逆扩散
    @torch.no_grad()
    def diff_eval_epoch(self,
                        valid_data=None,
                        show_progress=True):
        if not valid_data:
            return

        self.condition_model.eval()
        self.noise_model.eval()

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

        for batch_idx,batch in enumerate(iter_data):
            interaction, history_index, positive_u, positive_i = batch

            # NOTE: 1.CONDITION OUTPUT
            item_seq = interaction['item_id_list'].to(self.device)
            item_len = interaction['item_length'].to(self.device)
            cond_output = self.ddp_condition_model(item_seq, item_len)
            cond = {
                'y': cond_output.to(self.device)
            }

            # NOTE: 2.diffusion获得 xstart
            sample_shape = (cond_output.shape[0],self.rec_config['embedding_size'])
            sample = self.diffusion_model.p_sample_loop(
                self.noise_model,
                sample_shape,
                clip_denoised=self.diff_config.clip_denoised,
                model_kwargs=cond
            ) # 采样结果 xstart

            # NOTE: 3.最终推理
            assert cond_output.shape == sample.shape,'the condition output must have the same shape with diffusion sample.'
            final_sample = cond_output + sample

            test_items_emb = self.condition_model.item_embedding.weight
            scores = torch.matmul(final_sample,test_items_emb.transpose(0,1))

            if self.device.type == 'cuda' and show_progress:
                iter_data.set_postfix_str(
                    Reclogger.set_color("GPU RAM: " + Recutils.get_gpu_usage(self.device), "yellow")
                )

            self.eval_collector.eval_batch_collect(
                scores, interaction, positive_u, positive_i
            )

        # 指标计算
        struct = self.eval_collector.get_data_struct()
        valid_result = self.evaluator.evaluate(struct)
        valid_score = calculate_valid_score(valid_result, self.rec_config["valid_metric"].lower())
        return valid_result, valid_score


    # TODO: 重写training loop
    def train_loop_final(self,
                         train_data,
                         valid_data,
                         show_progress=True):

        # 1. 前一半训练gru

        for epoch_idx in range(0,self.epoch_size//2):
            train_loss = self.gru_train_epoch(train_data,epoch_idx,show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = train_loss

            # 记录loss
            train_loss_output = self._generate_train_loss_output_no_time(
                epoch_idx,train_loss
            )
            self.logger.info(train_loss_output)

            # 检验
            valid_result,valid_score = self.gru_eval_epoch(valid_data,show_progress=show_progress)

            # early stopping
            (
                self.best_cond_valid_results,
                self.cur_epoch,
                stop_flag,
                update_flag
            ) = early_stopping_results(
                valid_result,
                self.best_cond_valid_results,
                self.cur_epoch,
                max_step=self.stopping_step,
                bigger=self.rec_config['valid_metric_bigger']
            )
            # logging
            valid_result_output = (
                Reclogger.set_color('epoch %d evaluating','green')
                + Reclogger.set_color('  results: \n','blue')
            ) % (epoch_idx)
            valid_result_output += Recutils.dict2str(valid_result)
            valid_result_output += '\n'
            self.logger.info(valid_result_output)

            # 只save condition model
            if update_flag:
                cond_dict = dict(
                    config=self.rec_config,
                    model=self.condition_model
                )
                print('now is saving NO {} EPOCH'.format(epoch_idx))
                save_checkpoint(self.saved_model_path, condition_model_dict=cond_dict)
                # self.logger.info(
                #     Reclogger.set_color(f"Saving current epoch {epoch_idx}", "blue") + f": {self.saved_model_path}"
                # )
                self.best_cond_valid_results = deepcopy(valid_result)

            if stop_flag:
                stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_epoch * self.eval_step
                )
                self.logger.info(stop_output)
                print('stopping condition model training, the epoch idx is {}'.format(epoch_idx))
                break

            # return self.best_cond_valid_results

        # 2. 后一半训练diffusion
        self.cur_epoch = 0
        # 加载condition模型，加载best
        checkpoint = torch.load(os.path.join(self.saved_model_path,'condition_model.pt'))
        self.condition_model.load_state_dict(checkpoint['state_dict'])

        # 调学习率
        # 把优化器condition的优化关了
        for param_group in self.optimizer.param_groups:
            if "cond" in param_group.keys():
                param_group['lr'] = 0.0
                break

        for epoch_idx in range(self.epoch_size // 2,self.epoch_size):
            train_loss = self.diff_train_epoch(train_data,epoch_idx,show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = train_loss

            # 记录loss
            train_loss_output = self._generate_train_loss_output_no_time(
                epoch_idx, train_loss
            )
            self.logger.info(train_loss_output)

            valid_result,valid_score = self.diff_eval_epoch(valid_data,show_progress=show_progress)
            (
                self.best_valid_results,
                self.cur_epoch,
                stop_flag,
                update_flag
            ) = early_stopping_results(
                valid_result,
                self.best_valid_results,
                self.cur_epoch,
                max_step=self.stopping_step,
                bigger=self.rec_config['valid_metric_bigger']
            )

            # logging
            valid_result_output = (Reclogger.set_color('epoch %d evaluating', 'green')
                                          + Reclogger.set_color('  results: \n', 'blue')
                                  ) % (epoch_idx)
            valid_result_output += Recutils.dict2str(valid_result)
            valid_result_output += '\n'
            self.logger.info(valid_result_output)

            if update_flag:
                cond_dict = dict(
                    config=self.rec_config,
                    model=self.condition_model
                )
                print('now is saving NO {} EPOCH'.format(epoch_idx))
                save_checkpoint(self.saved_model_path, condition_model_dict=cond_dict,noise_model=self.noise_model)
                # self.logger.info(
                #     Reclogger.set_color(f"Saving current epoch {epoch_idx}", "blue") + f": {self.saved_model_path}"
                # )
                self.best_valid_results = deepcopy(valid_result)

            if stop_flag:
                stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_epoch * self.eval_step
                )
                self.logger.info(stop_output)
                print('stopping condition model training, the epoch idx is {}'.format(epoch_idx))
                break

        return self.best_valid_results

    # 没有gru预训练
    def train_loop_diff(self,
                   train_data,
                   valid_data,
                   show_progress=True):
        for epoch_idx in range(self.epoch_size):
            train_loss = self.diff_train_epoch(train_data,epoch_idx,show_progress = show_progress)
            self.train_loss_dict[epoch_idx] = train_loss

            # 记录loss
            train_loss_output = self._generate_train_loss_output_no_time(
                epoch_idx, train_loss
            )
            self.logger.info(train_loss_output)

            valid_result, valid_score = self.diff_eval_epoch(valid_data, show_progress=show_progress)
            (
                self.best_valid_results,
                self.cur_epoch,
                stop_flag,
                update_flag
            ) = early_stopping_results(
                valid_result,
                self.best_valid_results,
                self.cur_epoch,
                max_step=self.stopping_step,
                bigger=self.rec_config['valid_metric_bigger']
            )

            # logging
            valid_result_output = (Reclogger.set_color('epoch %d evaluating', 'green')
                                   + Reclogger.set_color('  results: \n', 'blue')
                                   ) % (epoch_idx)
            valid_result_output += Recutils.dict2str(valid_result)
            valid_result_output += '\n'
            self.logger.info(valid_result_output)

            if update_flag:
                cond_dict = dict(
                    config=self.rec_config,
                    model=self.condition_model
                )
                print('now is saving NO {} EPOCH'.format(epoch_idx))
                save_checkpoint(self.saved_model_path, condition_model_dict=cond_dict, noise_model=self.noise_model)
                # self.logger.info(
                #     Reclogger.set_color(f"Saving current epoch {epoch_idx}", "blue") + f": {self.saved_model_path}"
                # )
                self.best_valid_results = deepcopy(valid_result)

            if stop_flag:
                stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_epoch * self.eval_step
                )
                self.logger.info(stop_output)
                print('stopping condition model training, the epoch idx is {}'.format(epoch_idx))
                break
        self.best_cond_valid_results = {}
        return self.best_valid_results

    # NOTE: 检查gru用
    def train_loop_gru(self,
                   train_data,
                   valid_data,
                   show_progress=True):

        # TODO:loss的类型？更好的区分gru的loss和diffusion的loss
        best_valid_results = None
        for epoch_idx in range(0,self.epoch_size):
            # NOTE: 1.training
            train_loss = self.gru_train_epoch(train_data,epoch_idx,show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = train_loss

            # NOTE: 2. evaling
            valid_result,valid_score = self.gru_eval_epoch(valid_data,show_progress=show_progress)

            # NOTE: 3. early stopping && saving
            (
                best_valid_results,
                self.cur_epoch,
                stop_flag,
                update_flag,
            ) = early_stopping_results(
                valid_result,
                best_valid_results,
                self.cur_epoch,
                max_step=self.stopping_step,
                bigger=self.rec_config["valid_metric_bigger"],
            )

            # NOTE: 4.saving TODO:还得修改
            if update_flag:
                cond_dict = dict(
                    config=self.rec_config,
                    model=self.condition_model
                )
                print('now is saving NO {} EPOCH'.format(epoch_idx))
                save_checkpoint(self.saved_model_path, noise_model=self.noise_model, condition_model_dict=cond_dict)
                # self.logger.info(
                #     Reclogger.set_color(f"Saving current epoch {epoch_idx}", "blue") + f": {self.saved_model_path}"
                # )
                best_valid_results = deepcopy(valid_result)
                self.best_valid_results = deepcopy(valid_result)

            if stop_flag:
                # stop_output = "Finished training, best eval result in epoch %d" % (
                #         epoch_idx - self.cur_step * self.eval_step
                # )
                # if verbose:
                #     self.logger.info(stop_output)
                print('stopping epoch idx is {}'.format(epoch_idx))
                break

        self.best_cond_valid_results = self.best_valid_results
        return self.best_valid_results


    # NOTE: GRU训练函数
    def gru_train_epoch(self,train_data,epoch_idx,
                        loss_func='BPR',
                        show_progress=True):
        '''训练gru network'''
        # assert loss_func in ['BPR','CE'],'The type of loss function is unknown.'

        self.condition_model.train()

        total_losses = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=Reclogger.set_color(f"Train {epoch_idx}","pink"),
                # leave=False
            )
            if show_progress
            else train_data
        )

        if not self.rec_config["single_spec"] and train_data.shuffle:
            train_data.sampler.set_epoch(epoch_idx)

        scaler = GradScaler(enabled=self.enable_scaler)

        for batch_idx,batch in enumerate(iter_data):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # NOTE: gru的输出
            item_seq = batch['item_id_list']  # seq
            item_len = batch['item_length']  # len
            output = self.ddp_condition_model(item_seq, item_len)
            pos_items = batch['item_id']
            # NOTE: loss计算
            if loss_func == 'BPR':
                neg_items = batch['neg_item_id']
                neg_items_emb = self.condition_model.item_embedding(neg_items)
                pos_items_emb = self.condition_model.item_embedding(pos_items)

                loss = self.bpr_loss(output,pos_items_embed=pos_items_emb,neg_items_embed=neg_items_emb)
            elif loss_func == 'CE':
                loss = self.ce_loss(output,target_items=pos_items)
            else:
                raise NotImplementedError("Make sure 'loss_func' in ['BPR', 'CE']!")

            # NOTE:反向传播
            self._check_nan(loss)
            scaler.scale(loss).backward()
            total_losses = loss.item() if total_losses is None else total_losses + loss.item()

            # NOTE:优化
            scaler.step(self.optimizer)
            scaler.update()

            if self.device.type == 'cuda' and show_progress:
                output_dict = {'LOSS': total_losses, "GRU RAM": Recutils.get_gpu_usage(self.device)}
                iter_data.set_postfix(output_dict)
                # iter_data.set_postfix_str(
                #     Reclogger.set_color("LOSS: %.4f GPU RAM" + Recutils.get_gpu_usage(self.device), "yellow") % (total_losses)
                # )

        return total_losses

    # GRU的eval
    @torch.no_grad()
    def gru_eval_epoch(self,
                       valid_data=None,
                       show_progress=True):
        if not valid_data:
            return

        self.condition_model.eval()

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

        for batch_idx,batch in enumerate(iter_data):
            interaction, history_index, positive_u, positive_i = batch

            # gru输出
            item_seq = interaction['item_id_list'].to(self.device)
            item_len = interaction['item_length'].to(self.device)
            cond_output = self.ddp_condition_model(item_seq, item_len)

            # gru的推理
            test_items_emb = self.condition_model.item_embedding.weight
            scores = torch.matmul(
                cond_output, test_items_emb.transpose(0, 1)
            )

            if self.device.type == 'cuda' and show_progress:
                iter_data.set_postfix_str(
                    Reclogger.set_color("GPU RAM: " + Recutils.get_gpu_usage(self.device), "yellow")
                )

            self.eval_collector.eval_batch_collect(
                scores, interaction, positive_u, positive_i
            )

        # 指标计算
        struct = self.eval_collector.get_data_struct()
        valid_result = self.evaluator.evaluate(struct)
        valid_score = calculate_valid_score(valid_result, self.rec_config["valid_metric"].lower())
        return valid_result, valid_score


    def get_eps_xstart_from_forward(self,
                                    target_item,
                                    t,
                                    model_kwargs=None,
                                    denoised_fn=None):

        # 预测均值的获得
        def process_xstart(x):
            '''对x进行一定处理？'''
            if denoised_fn is not None:
                x = denoised_fn(x)
            if self.diff_config.clip_denoised:
                return x.clamp(-1, 1)
            return x

        # NOTE: 1. 扩散生成x_t
        if model_kwargs is None:
            model_kwargs = {}
        noise = torch.randn_like(target_item.float())
        x_t = self.diffusion_model.q_sample(x_start=target_item,t=t,noise=noise)

        # NOTE: 2. 获得神经网络的noise
        model_output = self.noise_model(x_t,
                                        self.diffusion_model._scale_timesteps(t),
                                        **model_kwargs)

        assert model_output.shape == noise.shape == target_item.shape

        # NOTE: 3. 获得x_0
        x_0 = process_xstart(
            self.diffusion_model._predict_xstart_from_eps(x_t=x_t,t=t,eps=model_output)
        )

        return {'noise':model_output,'xstart':x_0}


    # TODO: 直接神经网络获得x_0

    def _generate_train_loss_output_no_time(self,epoch_idx,losses):
        train_loss_output = (
            Reclogger.set_color('epoch %d training','green') + '['
        ) % (epoch_idx)

        if isinstance(losses,tuple):
            des = Reclogger.set_color('train_loss:%d','blue') + ": %.4f"
            train_loss_output += ','.join(
                des % (idx+1,loss) for idx,loss in enumerate(losses)
            )

        else:
            des = "%.4f"
            train_loss_output += Reclogger.set_color('train loss','blue') + ':' + des % losses


        return train_loss_output + "]"