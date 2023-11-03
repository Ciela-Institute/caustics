import torch
from torch import nn

from .stblocks import ISAB, PMA, SAB


class AlphaSetTransformerISAB(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dims_enc,
        hidden_dims_dec,
        out_dim,
        num_heads,
        num_inds,
        num_out=1,
        ln=False,
    ):
        super().__init__()

        enc_layers = [ISAB(in_dim, hidden_dims_enc[0], num_heads, num_inds, ln=ln)]
        for i in range(len(hidden_dims_enc) - 1):
            enc_layers.append(
                ISAB(
                    hidden_dims_enc[i],
                    hidden_dims_enc[i + 1],
                    num_heads,
                    num_inds,
                    ln=ln,
                )
            )

        self.enc = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = [
            PMA(hidden_dims_enc[-1], num_heads, num_out, ln=ln)
        ]
        hidden_dims_dec.insert(0, hidden_dims_enc[-1])
        for i in range(len(hidden_dims_dec) - 1):
            dec_layers.append(
                SAB(hidden_dims_dec[i], hidden_dims_dec[i + 1], num_heads, ln=ln)
            )

        dec_layers.append(nn.Linear(hidden_dims_dec[-1], out_dim))
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        enc_out = self.enc(x)
        dec_out = self.dec(enc_out).squeeze(dim=1)

        return dec_out


class AlphaSetTransformerSAB(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dims_enc,
        hidden_dims_dec,
        out_dim,
        num_heads,
        num_out=1,
        ln=False,
    ):
        super().__init__()

        enc_layers = [SAB(in_dim, hidden_dims_enc[0], num_heads, ln=ln)]
        for i in range(len(hidden_dims_enc) - 1):
            enc_layers.append(
                SAB(hidden_dims_enc[i], hidden_dims_enc[i + 1], num_heads, ln=ln)
            )

        self.enc = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = [
            PMA(hidden_dims_enc[-1], num_heads, num_out, ln=ln)
        ]
        hidden_dims_dec.insert(0, hidden_dims_enc[-1])
        for i in range(len(hidden_dims_dec) - 1):
            dec_layers.append(
                SAB(hidden_dims_dec[i], hidden_dims_dec[i + 1], num_heads, ln=ln)
            )

        dec_layers.append(nn.Linear(hidden_dims_dec[-1], out_dim))
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        enc_out = self.enc(x)
        dec_out = self.dec(enc_out).squeeze(dim=1)

        return dec_out


class ConvNet2D(nn.Module):
    def __init__(
        self, in_ch, chs, out_ch, activation, final_activation, wrapper_func=None
    ):
        super().__init__()

        if not wrapper_func:
            wrapper_func = lambda x: x

        chs.append(out_ch)

        layers = [
            nn.Conv2d(in_channels=in_ch, out_channels=chs[0], kernel_size=1, padding=0)
        ]

        for i in range(len(chs) - 1):
            layers.append(getattr(nn, activation)())
            layers.append(
                wrapper_func(
                    nn.Conv2d(
                        in_channels=chs[i],
                        out_channels=chs[i + 1],
                        kernel_size=1,
                        padding=0,
                    )
                )
            )
        assert layers[-1].bias is not None
        layers[-1].bias.data.fill_(0.0)

        if final_activation is not None:
            layers.append(getattr(nn, final_activation)())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: input Tensor, shape (b, c, npix, npix)
        """
        return self.net(x)


class JointModel_ST_CNN(nn.Module):
    def __init__(
        self,
        in_dim_SM,
        hidden_dims_enc_SM,
        hidden_dims_dec_SM,
        embedding_size,
        num_heads_SM,
        layernorm_SM,
        in_dim_MM,
        hidden_dims_MM,
        out_dim_MM,
        npix,
        normalizer,
        activation_MM="ReLU",
        final_activation_MM=None,
        num_inds_SM=None,
    ):
        super().__init__()

        self.npix = npix
        self.embedding_size = embedding_size
        self.normalizer = normalizer

        if num_inds_SM is not None:
            self.ST = AlphaSetTransformerISAB(
                in_dim_SM,
                hidden_dims_enc_SM,
                hidden_dims_dec_SM,
                embedding_size,
                num_heads_SM,
                num_inds_SM,
                ln=layernorm_SM,
            )
        else:
            self.ST = AlphaSetTransformerSAB(
                in_dim_SM,
                hidden_dims_enc_SM,
                hidden_dims_dec_SM,
                embedding_size,
                num_heads_SM,
                ln=layernorm_SM,
            )

        self.CNN = ConvNet2D(
            in_dim_MM, hidden_dims_MM, out_dim_MM, activation_MM, final_activation_MM
        )

    def forward(self, h, x):
        embedding = self.ST(h)
        embedding_ = (
            embedding.reshape(-1, self.embedding_size, 1, 1)
            .repeat_interleave(self.npix, dim=-2)
            .repeat_interleave(self.npix, dim=-1)
        )
        cnn_input = torch.cat([x, embedding_], dim=1)
        out = self.CNN(cnn_input)
        return out
