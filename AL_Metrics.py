import torch
import numpy as np

def entropy_sampling(stack):
    entropy_list = []
    for x in stack:
        unique_values, counts = torch.unique(x, return_counts=True)
        length = x.shape[0]
        if len(unique_values) == 1:
            entropy_list.append(0)
        else:
            entropy = 0
            for count in counts:
                entropy += count/length * np.log2(count/length)
            entropy_list.append(-entropy)
    return entropy_list

def margin_sampling(stack):
    margin_list = []
    for x in stack:
        unique_values, counts = torch.unique(x, return_counts=True)
        length = x.shape[0]
        if len(unique_values) == 1:
            margin_list.append(0)
        else:
            margin_list.append(counts[0]/length - counts[1]/length)
    return margin_list

def least_confidence(stack):
    least_confidence_list = []
    for x in stack:
        unique_values, counts = torch.unique(x, return_counts=True)
        length = x.shape[0]
        if len(unique_values) == 1:
            least_confidence_list.append(0)
        else:
            least_confidence_list.append(counts[0]/length)
    return least_confidence_list

def get_uncertainty(stack, metric):
    metric_dict = {"Least Confidence": least_confidence, "Margin Sampling": margin_sampling, "Entropy": entropy_sampling}
    return metric_dict[metric](stack)