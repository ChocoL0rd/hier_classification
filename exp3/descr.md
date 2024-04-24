Experiment 3
Uses class attention (cross attention with one token in output) https://arxiv.org/pdf/2103.17239.pdf.
Differ from previous sandbox in decoder implementation. No need for in embs, it returns only one its one 
embedding for each predict.
ViTCLSTree processes it to output of needed format due to pred_masks. 