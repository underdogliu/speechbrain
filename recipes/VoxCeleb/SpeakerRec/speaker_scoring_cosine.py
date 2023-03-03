#!/usr/bin/env python3
"""
Perform cosine scoring on two matrices (ok, enrol vs. test on voxceleb,
actually).

Note that in order to handle the situation where 
"""
import os
import logging
import sys

import numpy
import torch
from hyperpyyaml import load_hyperpyyaml

from speechbrain.utils.data_utils import download_file
from speechbrain.utils.metric_stats import EER, minDCF


def get_cosine_verification_scores(enrol_mat, test_mat, veri_test, params):
    """
    Compute the verification scores based on cosine similarity
    """
    scores = []
    positive_scores = []
    negative_scores = []

    save_file = os.path.join(params["output_folder"], "scores.txt")
    s_file = open(save_file, "w")

    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    # perform some checking and compute score
    assert enrol_mat.shape == test_mat.shape
    score = similarity(enrol_mat, test_mat)[0]

    # write score file
    for i, line in enumerate(veri_test):
        # Reading verification file (enrol_file test_file label)
        lab_pair = int(line.split(" ")[0].rstrip().split(".")[0].strip())
        enrol_id = line.split(" ")[1].rstrip().split(".")[0].strip()
        test_id = line.split(" ")[2].rstrip().split(".")[0].strip()

        # write score file
        s_file.write("%s %s %i %f\n" % (enrol_id, test_id, lab_pair, score))
        scores.append(score)

        if lab_pair == 1:
            positive_scores.append(score)
        else:
            negative_scores.append(score)

    s_file.close()
    return positive_scores, negative_scores


def get_enrol_mat(enrol_npy_file, veri_test):
    """
    This is a special function where we handle the case
        for normal ASV trials (not voxceleb)
    """
    pass


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))

    # Load hyperparameters file with command-line overrides
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])
    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)

    # Load verification list (to exlude verification sentences from train)
    veri_file_path = os.path.join(
        params["save_folder"], os.path.basename(params["verification_file"])
    )
    download_file(params["verification_file"], veri_file_path)
    logger.info("Loading verification trial list...")
    with open(veri_file_path) as f:
        veri_test = [line.rstrip() for line in f]

    # load and process enrolment and test data
    logger.info("Loading enrolment and testing data...")
    enrol_mat = numpy.load(sys.argv[1])
    test_mat = numpy.load(sys.argv[2])

    # Compute scores
    positive_scores, negative_scores = get_cosine_verification_scores(
        enrol_mat, test_mat, veri_test, params
    )

    # Compute statistics
    logger.info("Computing EER and minDCF...")
    eer, th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    logger.info("EER(%%)=%f", eer * 100)

    min_dcf, th = minDCF(
        torch.tensor(positive_scores), torch.tensor(negative_scores)
    )
    logger.info("minDCF=%f", min_dcf * 100)