import logging
import numpy as np
import torch
from utils import batch_cross_entropy, get_uncertainties
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_NAME = "bnn-project"


class ResultsLogger(object):

    def __init__(self,
                 exp_name,
                 model,
                 args):

        self.model = model

        wandb.init(
            name=exp_name,
            project=PROJECT_NAME,
            config=args,
        )

        wandb.watch(model, log="all")

        self.reset_metrics()

    def reset_metrics(self):
        self.train_metrics = {
            "complexity_costs": [],
            "likelihood_costs": [],
            "entropys": [],
            "epistemics": [],
            "aleatorics": [],
            "losses": [],
            "correct": 0,
            "total": 0,
        }

        self.test_metrics = {
            "complexity_costs": [],
            "likelihood_costs": [],
            "losses": [],
            "entropys": [],
            "epistemics": [],
            "aleatorics": [],
            "correct": 0,
            "total": 0,
            "examples": [],
        }

    def log_train_step(self,
                       data,
                       target,
                       pred,
                       batch_idx,
                       complexity_cost,
                       likelihood_cost,
                       loss,
                       train_loader):

        batch_size = len(data)
        dataset_size = len(train_loader.dataset)

        if batch_idx % 200 == 0:
            logger.info('{}/{} \tLoss: {:.6f}'
                        .format(
                            batch_idx * batch_size, dataset_size, loss.item()
                        ))

        self.train_metrics["complexity_costs"].append(complexity_cost.item())
        self.train_metrics["likelihood_costs"].append(likelihood_cost.item())
        self.train_metrics["losses"].append(loss.item())

        probs = torch.mean(torch.softmax(pred, dim=-1), dim=0)
        self.train_metrics["correct"] += (
            (torch.argmax(probs, 1) == target).sum().item()
        )
        self.train_metrics["total"] += len(data)

        entropys, epistemics, aleatorics = get_uncertainties(pred)
        self.train_metrics["entropys"].extend(entropys)
        self.train_metrics["epistemics"].extend(entropys)
        self.train_metrics["aleatorics"].extend(aleatorics)

    def log_test_step(self,
                      data,
                      target,
                      pred,
                      test_loader):

        dataset_size = len(test_loader.dataset)

        complexity_cost = self.model.kl_loss()
        likelihood_cost = batch_cross_entropy(pred, target, reduction='mean')
        loss = (1 / dataset_size) * complexity_cost + likelihood_cost

        self.test_metrics["complexity_costs"].append(complexity_cost.item())
        self.test_metrics["likelihood_costs"].append(likelihood_cost.item())
        self.test_metrics["losses"].append(loss.item())

        probs = torch.mean(torch.softmax(pred, dim=-1), dim=0)
        self.test_metrics["correct"] += (
            (torch.argmax(probs, 1) == target).sum().item()
        )
        self.test_metrics["total"] += len(data)

        entropys, epistemics, aleatorics = get_uncertainties(pred)
        self.test_metrics["entropys"].extend(entropys)
        self.test_metrics["epistemics"].extend(entropys)
        self.test_metrics["aleatorics"].extend(aleatorics)

        random_idx = random.randint(0, len(data)-1)
        example = (data[random_idx].cpu().numpy(),
                   target[random_idx].item(),
                   pred[:, random_idx, :].cpu().numpy())
        self.test_metrics["examples"].append(example)

    def log_epoch(self, epoch, train_time, test_time):

        test_accuracy = 100 * (self.test_metrics["correct"] /
                               self.test_metrics["total"])

        logger.info('Test set: Accuracy: {:.4f}'
                    .format(test_accuracy))

        wandb.log({
            "train_loss":
                np.mean(self.train_metrics["losses"]),
            "train_complexity_cost":
                np.mean(self.train_metrics["complexity_costs"]),
            "train_likelihood_cost":
                np.mean(self.test_metrics["likelihood_costs"]),
            "train_accuracy":
                self.train_metrics["correct"] / self.train_metrics["total"],
            "train_time": train_time,

            "test_loss":
                np.mean(self.test_metrics["losses"]),
            "test_complexity_cost":
                np.mean(self.test_metrics["complexity_costs"]),
            "test_likelihood_cost":
                np.mean(self.train_metrics["likelihood_costs"]),
            "test_accuracy":
                self.test_metrics["correct"] / self.test_metrics["total"],
            "test_time": test_time,

            "example_imgs":
                self.get_examples_figure(self.test_metrics["examples"]),

            "epoch": epoch+1
        })

        self.reset_metrics()

    def get_examples_figure(self, examples):
        rows = len(examples)
        fig = make_subplots(rows=rows, cols=2, column_widths=[0.3, 0.7])

        for i, example in enumerate(examples):
            data, target, pred = example
            img = np.squeeze(np.flip(data, axis=0))

            fig.add_trace(
                go.Heatmap(
                    z=img,
                    colorscale="Greys",
                    reversescale=True,
                    showscale=False,
                ),
                row=i+1,
                col=1
            )

            for cls, sampled_probs in enumerate(pred.T):
                fig.add_trace(
                    go.Violin(y=sampled_probs, name=cls, showlegend=False),
                    row=i+1,
                    col=2
                )

        return fig
