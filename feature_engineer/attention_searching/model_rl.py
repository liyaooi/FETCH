import torch.nn as nn
import torch
from .modules_rl import StatisticLearning, EncoderLayer, SelectOperations, ReductionDimension, weight_init
from torch.distributions.categorical import Categorical
import logging, os


class Actor(nn.Module):
    def __init__(self, args, data_nums, operations, d_model, d_k, d_v, d_ff, n_heads, dropout=None, enc_load_pth=None):
        super(Actor, self).__init__()
        self.args = args
        self.reduction_dimension = ReductionDimension(data_nums, d_model)
        self.encoder = EncoderLayer(d_model, d_k, d_v, d_ff, n_heads, dropout)
        logging.info(f"Randomly initial encoder")
        if os.path.exists(enc_load_pth):
            self.encoder.load_state_dict(torch.load(enc_load_pth))
            logging.info(f"Successfully load encoder, enc_load_pth:{enc_load_pth}")
        self.select_operation = SelectOperations(d_model, operations)
        self.c_nums = len(args.c_columns)
        self.layernorm = nn.LayerNorm(data_nums)

    def forward(self, input, step):
        input_norm = self.layernorm(input)
        data_reduction_dimension = self.reduction_dimension(input_norm)
        data_reduction_dimension = torch.where(torch.isnan(data_reduction_dimension),
                                               torch.full_like(data_reduction_dimension, 0), data_reduction_dimension)
        encoder_output = self.encoder(data_reduction_dimension)
        encoder_output = torch.where(torch.isnan(encoder_output), torch.full_like(encoder_output, 0), encoder_output)
        output = self.select_operation(encoder_output)
        output = torch.where(torch.isnan(output), torch.full_like(output, 0), output)
        operation_softmax = torch.softmax(output, dim=-1)

        return operation_softmax, data_reduction_dimension.squeeze(), \
               encoder_output.squeeze(), output
