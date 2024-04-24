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
        self.self_attn = nn.MultiheadAttention(
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

    def _sa_block(
            self,
            x,
            attn_mask: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
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
            x,
            attn_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
    ):
        x = x + self._sa_block(self.norm1(x), attn_mask, src_key_padding_mask)
        x = x + self._ffn_block(self.norm2(x))

        return x


class ClassDecoder(nn.Module):
    def __init__(
            self,
            depth,
            d_model,
            nhead,
            d_ff,
            dropout,
            activation,

            n_out_tokens
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.cls_token = nn.Parameter(torch.zeros((1, 1, d_model)), requires_grad=True)
        torch.nn.init.normal_(self.cls_token, std=.02)

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

    @staticmethod
    def generate_attention_mask(tgt_len, mem_len, device):
        # memory sees all memory
        mask_mem_mem = torch.zeros((mem_len, mem_len), device=device)  # (mem_len, mem_len)
        # memory doesn't see tgt
        mask_mem_tgt = torch.full((mem_len, tgt_len), float('-inf'), device=device)  # (mem_len, tgt_len)
        # memory sees
        mask_mem = torch.cat([mask_mem_mem, mask_mem_tgt], dim=-1)  # (mem_len, mem_len + tgt_len)

        # tgt sees only itself
        mask_tgt_tgt = torch.full((tgt_len, tgt_len), float('-inf'), device=device)  # # (tgt_len, tgt_len)
        torch.diagonal(mask_tgt_tgt, 0).fill_(0)
        # tgt sees all memory
        mask_tgt_mem = torch.zeros((tgt_len, mem_len), device=device)  # (tgt_len, mem_len)
        # tgt sees
        mask_tgt = torch.cat([mask_tgt_mem, mask_tgt_tgt], dim=-1)  # (tgt_len, mem_len + tgt_len)

        return torch.cat((mask_mem, mask_tgt), dim=0)  # (mem_len + tgt_len, mem_len + tgt_len)

    def forward(self, memory):
        b, l_mem, _ = memory.shape
        cls_tokens = self.cls_token.expand(b, 1, self.d_model)  # (B, 1, D)

        combined = torch.cat((memory, cls_tokens), dim=1)  # (B, L_mem+L_tgt, D)
        attn_mask = self.generate_attention_mask(1, l_mem, combined.device)

        for block in self.transformer:
            combined = block(combined, attn_mask=attn_mask)

        cls_tokens = combined[:, -1:, :]  # (B, 1, D)
        fin_tokens = self.fin_lin(cls_tokens)  # (B, 1, n_out_tokens)
        return fin_tokens


def configurator(
        cfg,
        n_out_tokens
):
    return ClassDecoder(
        depth=cfg["depth"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        activation=cfg["activation"],

        n_out_tokens=n_out_tokens,
    )
