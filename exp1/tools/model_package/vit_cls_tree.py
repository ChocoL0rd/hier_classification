import torch
from torch import nn
from .tokenizer import TreeTokenizer


class ViTCLSTree(nn.Module):
    def __init__(
            self,
            tokenizer: TreeTokenizer,
            encoder,
            decoder
    ):
        super().__init__()
        self.tokenizer = tokenizer

        self.encoder = encoder
        self.decoder = decoder

    def inference(self, img):
        """
            Iteratively returns each next attribute. Works only with one image
        :param img: (1, 3, H, W)
        :return: (L) list without cls, pad, eos.
        """
        # first predict of main class
        img_emb = self.encoder(img)
        device = img_emb.get_device()
        sent = torch.tensor([[self.tokenizer.mask_idx]], device=device)
        main_class = torch.argmax(self.forward_with_img_emb(sent, img_emb), dim=-1)  # (1)

        if not self.tokenizer.is_main_idx(main_class):
            return self.tokenizer.idxs2row(main_class.detach().cpu().numpy())

        n_attrs = self.tokenizer.get_n_attrs(int(main_class))

        sent = torch.cat(
            [
                main_class,
                torch.tensor([self.tokenizer.mask_idx for _ in range(n_attrs)]).to(device)
            ],
            dim=-1
        ).unsqueeze(0)  # (1, 1 + n_attrs)

        attributes = torch.argmax(self.forward_with_img_emb(sent, img_emb), dim=-1)  # (n_attrs)

        return self.tokenizer.idxs2row(torch.cat([main_class, attributes], dim=-1).detach().cpu().numpy())

    def forward_with_img_emb(self, sent, img_embs):
        """ See forward """
        is_pad_mask = self.tokenizer.idxs2is_pad_mask(sent)
        is_masked_mask = self.tokenizer.idxs2is_masked_mask(sent)
        pred = self.decoder(sent, img_embs, is_masked_mask, is_pad_mask)
        return pred

    def forward(self, sent, img):
        """
            For start training implementation
        :param sent: (B, L, D)
        :param img: (B, 3, H, W)
        :return: (B, n_tokens) what should be on the masked place
        """
        img_embs = self.encoder(img)  # (B, L, D)
        is_pad_mask = self.tokenizer.idxs2is_pad_mask(sent)
        is_masked_mask = self.tokenizer.idxs2is_masked_mask(sent)
        pred = self.decoder(sent, img_embs, is_masked_mask, is_pad_mask)

        return pred
