import torch
import torch.nn.functional as F


def cuda_timer(func, *args, **kwargs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    results = func(*args, **kwargs)
    end.record()

    torch.cuda.synchronize()

    cuda_time = start.elapsed_time(end) / 1000

    return results, cuda_time


def batch_linear(input, weight, bias):
    return torch.bmm(input, torch.transpose(weight, 1, 2)) + bias.unsqueeze(1)


def sampled_cross_entropies(input, target, reduction='mean'):
    # samples, batch_size, classes = input.shape

    # # flattening across samples & batches
    # flattened_input = input.flatten(0, 1)

    # # Repeating targets to match number of samples
    # repeated_target = target.repeat(samples)

    # batch_cross_entropy =  F.cross_entropy(
    #     flattened_input,
    #     repeated_target,
    #     reduction=reduction)

    sampled_cross_entropies = []
    for sample in input:
        sampled_cross_entropies.append(
            F.cross_entropy(sample, target, reduction=reduction)
        )

    sampled_cross_entropies = torch.stack(sampled_cross_entropies)

    # return batch_cross_entropy
    return sampled_cross_entropies


def get_uncertainties(logits):
    if not logits.dim() == 3:
        raise ValueError

    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)

    # Entropy of mean probs across samples
    entropy = torch.mean(
        torch.mean(probs, dim=0) * -torch.mean(log_probs, dim=0),
        dim=-1
    )

    # Aleatoric uncertainty is given by the mean of Entropies
    aleatoric = torch.mean(
        torch.mean(probs * -log_probs, dim=-1),
        dim=0
    )

    epistemic = entropy - aleatoric

    return (entropy,
            epistemic,
            aleatoric)
