import numpy as np
import torch
import torch.nn.functional as F

from torch import nn


def get_non_pad_mask(seq, pad_idx):
    assert seq.dim() == 2
    return seq.ne(pad_idx).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q, pad_idx):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


class Encoder(nn.Module):
    def __init__(self, model_config):
        super(Encoder, self).__init__()

        self.model_config = model_config
        
        len_max_seq = model_config["max_seq_len"]
        n_position = len_max_seq + 1
        n_layers = model_config["encoder_layer"]

        self.src_word_emb = nn.Embedding(
            model_config["vocab_size"],
            model_config["encoder_dim"],
            padding_idx=model_config["pad_idx"]
        )

        self.position_enc = nn.Embedding(
            n_position,
            model_config["encoder_dim"],
            padding_idx=model_config["pad_idx"]
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            model_config=model_config,
            d_model=model_config["encoder_dim"],
            d_inner=model_config["encoder_conv1d_filter_size"],
            n_head=model_config["encoder_head"],
            d_k=model_config["encoder_dim"] // model_config["encoder_head"],
            d_v=model_config["encoder_dim"] // model_config["encoder_head"],
            dropout=model_config["encoder_dropout"]
        ) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):
        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(
            seq_k=src_seq, seq_q=src_seq,
            pad_idx=self.model_config["pad_idx"]
        )
        non_pad_mask = get_non_pad_mask(
            seq=src_seq,
            pad_idx=self.model_config["pad_idx"]
        )
        
        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        
        return enc_output, non_pad_mask


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, model_config):
        super(Decoder, self).__init__()

        self.model_config = model_config

        len_max_seq = model_config["max_seq_len"]
        n_layers = model_config["decoder_layer"]

        self.position_enc = nn.Embedding(
            len_max_seq + 1,
            model_config["decoder_dim"],
            padding_idx=model_config["pad_idx"],
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            model_config=model_config,
            d_model=model_config["decoder_dim"],
            d_inner=model_config["decoder_conv1d_filter_size"],
            n_head=model_config["decoder_head"],
            d_k=model_config["decoder_dim"] // model_config["decoder_head"],
            d_v=model_config["decoder_dim"] // model_config["decoder_head"],
            dropout=model_config["decoder_dropout"]
        ) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):
        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(
            seq_k=enc_pos, seq_q=enc_pos,
            pad_idx=self.model_config["pad_idx"]
        )
        non_pad_mask = get_non_pad_mask(
            seq=enc_pos,
            pad_idx=self.model_config["pad_idx"]
        )

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self,
                 model_config,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            model_config, d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # q, k, v: [ (batch_size * n_heads) x seq_len x hidden_size ]

        d = np.sqrt(q.shape[-1])
        attn = F.softmax(q @ k.transpose(-2, -1) / d, dim=-1)
        # attn: [ (batch_size * n_heads) x seq_len x seq_len ]

        output = attn @ v
        # output: [ (batch_size * n_heads) x seq_len x hidden_size ]
        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(
            temperature=d_k**0.5) 
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
         # normal distribution initialization better than kaiming(default in pytorch)
        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_v))) 
        
    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv
        
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, model_config, d_in, d_hid, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=model_config["fft_conv1d_kernel"][0],
            padding=model_config["fft_conv1d_padding"][0]
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=model_config["fft_conv1d_kernel"][1],
            padding=model_config["fft_conv1d_padding"][1]
        )

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output

