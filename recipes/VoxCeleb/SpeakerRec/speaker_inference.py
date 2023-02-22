#!/usr/bin/python3
"""
Recipe for computing the speaker embeddings and store them in
various manner. 
This recipe is designed to work on a single GPU.

To run this recipe, run the following command:
    > python3 speaker_inference.py enroll_list.txt enroll.npy
"""
import os
import sys
import logging

import numpy
import torch
import torchaudio
import speechbrain as sb
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml

from speechbrain.utils.distributed import run_on_main

# Compute embeddings from the waveforms
def compute_embedding(wavs, wav_lens, params):
    """Compute speaker embeddings.

    Arguments
    ---------
    wavs : Torch.Tensor
        Tensor containing the speech waveform (batch, time).
        Make sure the sample rate is fs=16000 Hz.
    wav_lens: Torch.Tensor
        Tensor containing the relative length for each sentence
        in the length (e.g., [0.8 0.6 1.0])
    """
    with torch.no_grad():
        feats = params["compute_features"](wavs)
        feats = params["mean_var_norm"](feats, wav_lens)
        embeddings = params["embedding_model"](feats, wav_lens)
    return embeddings.squeeze(1)


def compute_embedding_loop(input_list, data_folder, output_file):
    """compute speaker embeddings from the file
    and store them into a matrix.
    Note that the matrix is listed w.r.t to the
        list order, so it is better to not shuffle
        anything.
    """
    embeddings = []
    num_files = 0
    with open(input_list, "r") as inl:
        for line in inl:
            wav_path = data_folder + "/" + line.split()[0].rstrip()
            wav, _ = torchaudio.load(wav_path)
            data = wav.transpose(0, 1).squeeze(1).unsqueeze(0)
            embedding = compute_embedding(data, torch.Tensor([data.shape[0]]), params).squeeze()
            embeddings.append(embedding)
            num_files += 1
    
    embeddings = torch.stack(embeddings).numpy()
    numpy.save(output_file, embeddings, allow_pickle=True)
    print("x-vector for {} files from {}, are stored in {}".format(num_files, input_list, output_file))
    

if __name__ == "__main__":

    # load directories
    input_list = sys.argv[1]
    output_file = sys.argv[2]

    # Logger setup
    logger = logging.getLogger(__name__)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))

    # Load hyperparameters file with command-line overrides
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[3:])
    print(params_file)
    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides=None)
    
    # Initiate/find experimental directory
    exp_folder = params["save_folder"]
    output_file = exp_folder + "/" + output_file

    # Load model
    run_on_main(params["pretrainer"].collect_files)
    params["pretrainer"].load_collected(params["device"])
    params["embedding_model"].eval()
    params["embedding_model"].to(params["device"])

    # Perform embedding extraction
    data_folder = params["data_folder"]
    compute_embedding_loop(input_list, data_folder, output_file)