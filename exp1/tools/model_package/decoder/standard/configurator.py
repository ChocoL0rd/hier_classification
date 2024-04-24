import torch
from torch import nn


class StandardDecoder(nn.Module):
    def __init__(
            self,
            depth,
            d_model,
            nhead,
            d_ff,
            dropout,
            activation,

            sent_length,
            n_tokens,
            pad_idx
    ):
        super().__init__()
        """
        :param depth: number of decoder blocks
        :param d_model: dimension of tokens, decoder operate with
        :param sent_length: number of pos embeddings including cls and eos
        :param n_tokens: number of tokens to use including mask, padding, cls, eos
        :param pad_idx: index of padding to don't count the gradient
        """
        self.d_model = d_model
        self.nhead = nhead
        self.n_tokens = n_tokens
        self.emb = nn.Embedding(
            num_embeddings=n_tokens,
            embedding_dim=d_model,
            padding_idx=pad_idx
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, sent_length, d_model),
            requires_grad=True
        )
        torch.nn.init.normal_(self.pos_embed, std=.02)

        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=depth,
        )

        self.fin_lin = nn.Linear(d_model, n_tokens)

    def create_causal_mask(self, b, l):
        """
            The purpose is to create mask that allows all not first tokens
        look only on the first token and itself
        :param b: batch_size
        :param l: sentence length
        :return: (B * nhead, L)
        """

        batch_causal_mask = torch.eye(l).unsqueeze(0).expand(b, l, l)
        batch_causal_mask[:, :, 0] = 1
        batch_causal_mask = batch_causal_mask == 0

        causal_mask = (
            batch_causal_mask.
            unsqueeze(1).  # (B, 1, L, L)
            expand(b, self.nhead, l, l).  # (B, nhead, L, L)
            reshape(b * self.nhead, l, l)  # (B * nhead, L, L)
        )

        return causal_mask

    def forward(self, tgt, memory, is_masked_mask, is_pad_mask):
        """
        :param tgt: (B, L)
        :param memory: (B, L1, D)
        :param is_masked_mask: (B, L) - True - mask token, False - not
        :param is_pad_mask: (B, L), True - pad token, False - not
        :return: tensor of shape (B, n_tokens)
        """
        b, l = tgt.shape
        tgt_tokens = self.emb(tgt) + self.pos_embed[:, :l].expand(b, l, self.d_model)

        tgt_tokens = self.transformer(
            tgt_tokens,
            memory,
            tgt_mask=self.create_causal_mask(b, l).to(tgt.get_device()),
            tgt_is_causal=True,
            tgt_key_padding_mask=is_pad_mask
        )  # (B, L, D)

        fin_tokens = self.fin_lin(tgt_tokens[is_masked_mask])
        return fin_tokens


def configurator(
        cfg,
        sent_length,
        n_tokens,
        pad_idx
):
    return StandardDecoder(
        depth=cfg["depth"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        activation=cfg["activation"],

        sent_length=sent_length,
        n_tokens=n_tokens,
        pad_idx=pad_idx
    )
