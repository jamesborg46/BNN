import torch
import numpy as np
from utils import sampled_cross_entropies, get_uncertainties


def train(model,
          device,
          train_loader,
          optimizer,
          results_logger,
          epoch,
          samples=1):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        dataset_size = len(train_loader.sampler.indices)
        optimizer.zero_grad()
        logits = model(data.flatten(1), samples=samples)
        complexity_cost = model.complexity_cost()
        likelihood_cost = sampled_cross_entropies(
            logits, target, reduction='mean')
        sampled_losses = (
            (1 / (dataset_size)) * complexity_cost + likelihood_cost
        )
        model.propagate_loss(sampled_losses)
        optimizer.step()

        loss = torch.mean(sampled_losses)
        results_logger.log_train_step(
            data,
            target,
            logits,
            batch_idx,
            torch.mean(complexity_cost),
            torch.mean(likelihood_cost),
            loss,
            train_loader
        )


def test(model, device, test_loader, results_logger, samples=30):
    model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data.flatten(1), samples=30)

            results_logger.log_test_step(
                data,
                target,
                logits,
                test_loader
            )


def active_sampling(model,
                    device,
                    train_data,
                    train_loader,
                    results_logger,
                    active_samples,
                    samples=30):

    model.eval()
    with torch.no_grad():
        print("Length before active sampling {}".format(
            len(train_loader.sampler.indices)))
        remaining_indices = np.array(list(
            set(range(len(train_data))) - set(train_loader.sampler.indices)
        ))

        remaining_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_data, remaining_indices),
            shuffle=False,
            batch_size=5000,
            num_workers=4,
        )

        batch_epistemics = []
        for data, _ in remaining_loader:
            data = data.to(device)
            logits = model(data.flatten(1), samples)
            entropys, epistemics, aleatorics = get_uncertainties(logits)
            batch_epistemics.append(epistemics)

        batch_epistemics = torch.cat(batch_epistemics)
        _, idxs = torch.topk(batch_epistemics, active_samples)

        new_indices = np.concatenate(
            [train_loader.sampler.indices,
             remaining_indices[idxs.cpu().numpy()]])

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=train_loader.batch_size,
            num_workers=train_loader.num_workers,
            sampler=torch.utils.data.SubsetRandomSampler(new_indices)
        )

        print("Length after active sampling {}".format(
            len(train_loader.sampler.indices)))

    return train_loader

