import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from layers.Autoformer_EncDec import series_decomp
from layers.Conv_Blocks import Inception_Block_V2


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class DLinearBlock(nn.Module):
    """Lightweight DLinear block for baseline forecasting."""
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.decomp = series_decomp(configs.moving_avg)
        self.linear_season = nn.Linear(self.seq_len, self.pred_len)
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
        self.linear_season.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        self.linear_trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def forward(self, x):
        season, trend = self.decomp(x)
        season_out = self.linear_season(season.permute(0, 2, 1))
        trend_out = self.linear_trend(trend.permute(0, 2, 1))
        out = season_out + trend_out
        return out.permute(0, 2, 1)


class TimesBlockLite(nn.Module):
    """A lightweight version of TimesBlock for fast inference."""
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = max(1, configs.top_k // 2)
        kernel_num = max(2, configs.num_kernels // 2)
        self.conv = nn.Sequential(
            Inception_Block_V2(configs.d_model, configs.d_ff, num_kernels=kernel_num),
            nn.GELU(),
            Inception_Block_V2(configs.d_ff, configs.d_model, num_kernels=kernel_num)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.seq_len + self.pred_len) // period + 1) * period
                padding = torch.zeros([x.shape[0], length - (self.seq_len + self.pred_len), x.shape[2]], device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.seq_len + self.pred_len, :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        return res + x


class Model(nn.Module):
    """TimesNet variant with interval prediction."""
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.interval_mult = getattr(configs, 'interval_mult', 2.0)

        self.baseline = DLinearBlock(configs)
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        # project encoder output to include the prediction horizon so the
        # lightweight TimesBlocks can operate on both history and future
        self.predict_linear = nn.Linear(self.seq_len, self.seq_len + self.pred_len)
        self.layers = nn.ModuleList([TimesBlockLite(configs) for _ in range(configs.e_layers)])
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.mean_projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        self.var_projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        # DLinear baseline provides a coarse forecast over the prediction horizon
        baseline = self.baseline(x_enc)

        # encode historical sequence and extend to cover the prediction horizon
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        # lightweight residual modeling
        for block in self.layers:
            enc_out = self.layer_norm(block(enc_out))

        mean_residual = self.mean_projection(enc_out)
        log_var = self.var_projection(enc_out)

        # combine baseline with residual on the prediction window only
        mean = baseline + mean_residual[:, -self.pred_len:, :]
        std = torch.sqrt(F.softplus(log_var[:, -self.pred_len:, :]) + 1e-6)
        lower = mean - self.interval_mult * std
        upper = mean + self.interval_mult * std

        # aggregate statistics over the whole prediction range instead of
        # returning point-wise forecasts
        mean_agg = mean.mean(dim=1)
        lower_agg = lower.min(dim=1).values
        upper_agg = upper.max(dim=1).values

        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return mean_agg, lower_agg, upper_agg
        else:
            return mean_agg
