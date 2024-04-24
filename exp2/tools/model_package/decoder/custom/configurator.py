import torch
from torch import nn, Tensor
from typing import Optional


def _get_activation(act):
    if isinstance(act, str):
        if act == "relu":
            return nn.ReLU()
        elif act == "gelu":
            return nn.GELU()

    raise Exception(f"activation must be relu or gelu, not {act}")


class CustomDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            d_ff: int = 2048,
            dropout: float = 0.1,
            activation: str = "gelu",
            layer_norm_eps: float = 1e-5,
            bias: bool = True,
    ):
        super().__init__()

        self.activation = _get_activation(activation)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
            bias=bias
        )

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout2 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model, bias=bias)

    def _ca_block(
            self,
            x, mem,
            attn_mask: Optional[Tensor],
            key_padding_mask: Optional[Tensor],
            is_causal: bool = False
    ) -> Tensor:
        x = self.cross_attn(
            x, mem, mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False
        )[0]
        return self.dropout1(x)

    def _ffn_block(self, x: Tensor) -> Tensor:
        x = self.linear2(
            self.dropout(
                self.activation(
                    self.linear1(x)
                )
            )
        )
        return self.dropout2(x)

    def forward(
            self,
            x, memory,
            memory_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            memory_is_causal: bool = False,
    ):
        x = x + self._ca_block(self.norm1(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
        x = x + self._ffn_block(self.norm2(x))

        return x


class CustomDecoder(nn.Module):
    def __init__(
            self,
            depth,
            d_model,
            nhead,
            d_ff,
            dropout,
            activation,

            n_in_tokens,
            n_out_tokens,
            pad_idx
    ):
        super().__init__()
        """
            CustomDecoder - uses only cross attention and ffn
        :param depth: number of decoder blocks
        :param d_model: dimension of tokens, decoder operate with
        :param sent_length: number of pos embeddings including cls and eos
        :param n_tokens: number of tokens to use including mask, padding, cls, eos
        :param pad_idx: index of padding to don't count the gradient
        """
        self.d_model = d_model
        self.nhead = nhead
        self.n_in_tokens = n_in_tokens
        self.in_emb = nn.Embedding(
            num_embeddings=n_in_tokens,
            embedding_dim=d_model,
            padding_idx=pad_idx
        )

        self.transformer = nn.ModuleList(
            [
                CustomDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for _ in range(depth)
            ]
        )

        self.fin_lin = nn.Linear(d_model, n_out_tokens)

    def forward(self, tgt, memory, is_pad_mask):
        """
        :param tgt: (B, L)
        :param memory: (B, L1, D)
        :param is_pad_mask: (B, L), True - pad token, False - not
        :return: tensor of shape (B, n_tokens)
        """
        b, l = tgt.shape
        tgt_tokens = self.in_emb(tgt)

        for block in self.transformer:
            tgt_tokens = block(tgt_tokens, memory)  # (B, L, D)

        fin_tokens = self.fin_lin(tgt_tokens[~is_pad_mask])
        return fin_tokens


def configurator(
        cfg,
        n_in_tokens,
        n_out_tokens,
        pad_idx
):
    return CustomDecoder(
        depth=cfg["depth"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        activation=cfg["activation"],

        n_in_tokens=n_in_tokens,
        n_out_tokens=n_out_tokens,
        pad_idx=pad_idx
    )
