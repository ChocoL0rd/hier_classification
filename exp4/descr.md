Experiment 4
As C-Tran and experiment 2
Decoder is implemented as decoder only transformer paradigm with images. 
Image embeddings concatenated with tokens and go through transformer blocks together.
Attention mask implemented so:
image embeddings see all image embeddings
each token sees only itself and all image embeddings

