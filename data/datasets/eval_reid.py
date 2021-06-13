# encoding: utf-8
import numpy as np


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50,q_ambis=None,g_ambis=None):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis])#.astype(np.int32)

    new_eval = (q_ambis is not None) and (g_ambis is not None)
    if new_eval:
        matches_am2id = (g_ambis[indices] == q_pids[:, np.newaxis])
        matches_id2am = (g_pids[indices] == q_ambis[:, np.newaxis])
        matches_am2am = (g_ambis[indices] == q_ambis[:, np.newaxis])
        matches = matches | matches_am2am | matches_am2id | matches_id2am
    matches = matches.astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_camids[order] == q_camid)
        if not new_eval:
            remove = remove & (g_pids[order] == q_pid)
        else:
            q_amb = q_ambis[q_idx]
            remove_dis = remove & (g_pids[order] == 0) # distractor with same cam
            remove_id2id = remove & (g_pids[order] == q_pid) 
            remove_am2id = remove & (g_ambis[order] == q_pid) 
            remove_am2am = remove & (g_ambis[order] == q_amb) 
            remove_id2am = remove & (g_pids[order] == q_amb) 
            remove = remove_dis | remove_id2id | remove_am2id | remove_am2am | remove_id2am

        # remove = remove | (g_pids[order] == -1)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP
