Experiment 1

Modified image2text, where attributes are computed in parallel. No [EOS]. 

StandardDecoder.create_causal_mask and VitCLSTree.inference are changed to implememt parallel processing of attributes.
Also in CLSTreeTokenizer get_n_attr added by main class. 