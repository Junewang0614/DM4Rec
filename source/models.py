from abc import abstractmethod
import torch
import torch.nn as nn

from tool_nn import GroupNorm32,timestep_embedding

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class ResBlock(TimestepBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 emb_channels,
                 dropout=0.0,
                 use_scale_shift_norm=False):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels # x_t的输入大小
        self.emb_channels = emb_channels # timestep的输入大小
        self.out_channels = out_channels
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm

        # 对xt,先不添加再编码的层了
        # TODO:self.in_layers
        # TODO:normalization layer
        self.in_layers = nn.Sequential(
            GroupNorm32(32, in_channels),
            nn.SiLU(),
        )

        # 对时间的编码
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_channels,
                      out_features=2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                      )
        )

        # 最后的输出层
        self.out_layers = nn.Sequential(
            GroupNorm32(32,out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout)
        )

        assert in_channels == out_channels,'input embedding is different with output embedding'
        self.skip_connection = nn.Identity()

    def forward(self, x, emb):
        '''
        :param x: [b,in_channels]
        :param emb: [b,emb_channels]
        :return: [b,out_channels]
        usually in_channels == out_channels
        '''
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)  # 这里把时间embedding拆成了两份，所以时间的embedding是图片通道数的两倍
            h = out_norm(h) * (1 + scale) + shift # 换了种方式正则化
            h = out_rest(h)
        else:
            h = h + emb_out  # embedding成相同大小以后直接相加
            h = self.out_layers(h)

        return self.skip_connection(x) + h

class MultiLayerPNet(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes:list,
                 time_emb_size = 32,
                 act_fun = 'relu',
                 class_cond = False,
                 cond_embed = -1):
        super(MultiLayerPNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.time_embed_size = time_emb_size
        self.class_cond = class_cond # 是否引入条件

        # 构造层
        # NOTE:1. 对时间步的编码层
        time_dim = 4 * time_emb_size
        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_size,time_dim),
            nn.SiLU(),
            nn.Linear(time_dim,time_dim)
        )

        # TODO:2.添加标签层num_classes
        # TODO 这里的网络应该是个RNN,用GRU4rec的结果
        # NOTE: 要求，保证最后embedding的形状是 [batch_size,time_dim]
        assert (cond_embed > 0) == class_cond, 'when class_cond is true,the class_cond is needed to have + number'
        if self.class_cond:
            self.cond_emb = nn.Linear(cond_embed, time_dim)

        # NOTE:3. 模型部分
        # NOTE: 结构：一层linear，一层activate,一层res
        # TODO: activate改成可以选择的

        self.hidden_sizes = hidden_sizes + [output_size]
        # 第一层
        self.network = nn.ModuleList(
            [TimestepEmbedSequential(
                nn.Linear(self.input_size,self.hidden_sizes[0]),
                nn.ReLU(),
                ResBlock(self.hidden_sizes[0],self.hidden_sizes[0],time_dim),
            )]
        )

        # 后面的层
        if len(self.hidden_sizes) > 1:
            for i in range(len(self.hidden_sizes)-1):
                self.network.append(
                    TimestepEmbedSequential(
                        nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]),
                        nn.ReLU(),
                        ResBlock(self.hidden_sizes[i+1], self.hidden_sizes[i+1], time_dim),
                    )
                )

        #NOTE:3. output层
        self.output = nn.Sequential(
            GroupNorm32(32,self.output_size),
            nn.SiLU(), # ?
        )

    def forward(self,x,timesteps,y=None):

        assert (y is not None) == self.class_cond,"must specify y if and only if the model is class-conditional"

        emb = self.time_embed(timestep_embedding(timesteps,self.time_embed_size))
        if self.class_cond is not None:
            y = self.cond_emb(y)
            assert y.shape[0] == x.shape[0]
            emb = emb + y

        h = x
        for module in self.network:
            h = module(h,emb)
        out = self.output(h)
        return out

