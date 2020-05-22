import torch
from utils import batch_cross_entropy


def train(model, device, train_loader, optimizer, results_logger, samples=1):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        dataset_size = len(train_loader.dataset)

        optimizer.zero_grad()
        pred = model(data.flatten(1), samples=samples)
        complexity_cost = model.kl_loss()
        likelihood_cost = batch_cross_entropy(pred, target, reduction='mean')
        loss = (1 / dataset_size) * complexity_cost + likelihood_cost
        loss.backward()
        optimizer.step()

        results_logger.log_train_step(
            data,
            target,
            pred,
            batch_idx,
            complexity_cost,
            likelihood_cost,
            loss,
            train_loader
        )


def test(model, device, test_loader, results_logger, samples=30):
    model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data.flatten(1), samples=30)

            results_logger.log_test_step(
                data,
                target,
                pred,
                test_loader
            )
