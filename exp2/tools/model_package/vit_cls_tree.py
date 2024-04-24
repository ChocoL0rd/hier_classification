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
        self.pred_mask = torch.tensor(tokenizer.pred_mask)
        self.encoder = encoder
        self.decoder = decoder

    def inference(self, img):
        """
            Iteratively returns each next attribute. Works only with one image
        :param img: (1, 3, H, W)
        :return: (L) list of predicts
        """
        img_emb = self.encoder(img)
        device = img_emb.get_device()

        # first predict of main class
        sent = torch.tensor([[0]], device=device)
        main_class = torch.argmax(self.forward_with_img_emb(sent, img_emb), dim=-1)  # (1)

        if not self.tokenizer.is_main_idx(main_class):
            return self.tokenizer.idxs2row(main_class.detach().cpu().numpy())

        attr_mask_idxs = torch.tensor(
            self.tokenizer.attrs_mask_tree[self.tokenizer.idx2name[main_class.item()]]
        ).unsqueeze(0).to(device)
        attributes = torch.argmax(
            self.forward_with_img_emb(attr_mask_idxs, img_emb),
            dim=-1
        )
        return self.tokenizer.idxs2row(torch.cat([main_class, attributes], dim=-1).detach().cpu().numpy())

    def forward_with_img_emb(self, sent, img_embs):
        """ See forward """
        is_pad_mask = self.tokenizer.idxs2is_pad_mask(sent)
        pred = self.decoder(sent, img_embs, is_pad_mask)
        pred_mask = self.pred_mask.to(pred.get_device())
        pred_mask = pred_mask[sent][~is_pad_mask]

        return pred + pred_mask

    def forward(self, sent, img):
        """
            For start training implementation
        :param sent: (B, L, D)
        :param img: (B, 3, H, W)
        :return: (B, n_tokens) what should be on the masked place
        """
        img_embs = self.encoder(img)  # (B, L, D)
        is_pad_mask = self.tokenizer.idxs2is_pad_mask(sent)

        pred = self.decoder(sent, img_embs, is_pad_mask)

        pred_mask = self.pred_mask.to(pred.get_device())
        pred_mask = pred_mask[sent][~is_pad_mask]

        return pred + pred_mask
