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
            "correct": [],
            "num_correct": 0,
            "total": 0,
        }

        self.test_metrics = {
            "complexity_costs": [],
            "likelihood_costs": [],
            "losses": [],
            "entropys": [],
            "epistemics": [],
            "aleatorics": [],
            "correct": [],
            "num_correct": 0,
            "total": 0,
            "examples": [],
        }

    def log_train_step(self,
                       data,
                       target,
                       logits,
                       batch_idx,
                       complexity_cost,
                       likelihood_cost,
                       loss,
                       train_loader):

        with torch.no_grad():
            batch_size = len(data)
            dataset_size = len(train_loader.sampler.indices)

            if batch_idx % 200 == 0:
                logger.info('{}/{} \tLoss: {:.6f}'
                            .format(
                                batch_idx * batch_size,
                                dataset_size,
                                loss.item()
                            ))

            self.train_metrics["complexity_costs"].append(
                complexity_cost.item()
            )
            self.train_metrics["likelihood_costs"].append(
                likelihood_cost.item()
            )
            self.train_metrics["losses"].append(loss.item())
            self.train_metrics["training_size"] = dataset_size

            mean_probs = torch.mean(torch.softmax(logits, dim=-1), dim=0)
            preds = torch.argmax(mean_probs, 1)

            self.train_metrics["correct"].append(preds == target)
            self.train_metrics["num_correct"] += (
                (preds == target).sum().item()
            )
            self.train_metrics["total"] += len(data)

            entropys, epistemics, aleatorics = get_uncertainties(logits)
            self.train_metrics["entropys"].append(entropys)
            self.train_metrics["epistemics"].append(epistemics)
            self.train_metrics["aleatorics"].append(aleatorics)

    def log_test_step(self,
                      data,
                      target,
                      logits,
                      test_loader):

        with torch.no_grad():
            dataset_size = len(test_loader.dataset)

            complexity_cost = self.model.kl_loss()
            likelihood_cost = batch_cross_entropy(logits,
                                                  target,
                                                  reduction='mean')
            loss = (1 / dataset_size) * complexity_cost + likelihood_cost

            self.test_metrics["complexity_costs"].append(
                complexity_cost.item()
            )
            self.test_metrics["likelihood_costs"].append(
                likelihood_cost.item()
            )
            self.test_metrics["losses"].append(loss.item())

            probs = torch.softmax(logits, dim=-1)
            mean_probs = torch.mean(probs, dim=0)
            preds = torch.argmax(mean_probs, 1)

            self.test_metrics["correct"].append(preds == target)
            self.test_metrics["num_correct"] += (
                (preds == target).sum().item()
            )
            self.test_metrics["total"] += len(data)

            entropys, epistemics, aleatorics = get_uncertainties(logits)
            self.test_metrics["entropys"].append(entropys)
            self.test_metrics["epistemics"].append(epistemics)
            self.test_metrics["aleatorics"].append(aleatorics)

            random_idx = random.randint(0, len(data)-1)
            example = (data[random_idx].cpu().numpy(),
                       target[random_idx].item(),
                       probs[:, random_idx, :].cpu().numpy())
            self.test_metrics["examples"].append(example)

    def log_epoch(self, epoch, train_time, test_time):

        with torch.no_grad():
            test_accuracy = 100 * (self.test_metrics["num_correct"] /
                                   self.test_metrics["total"])

            logger.info('Test set: Accuracy: {:.4f}'
                        .format(test_accuracy))

            test_entropys = torch.cat(self.test_metrics["entropys"])
            test_epistemics = torch.cat(self.test_metrics["epistemics"])
            test_aleatorics = torch.cat(self.test_metrics["aleatorics"])
            test_correct = torch.cat(self.test_metrics["correct"])

            train_entropys = torch.cat(self.train_metrics["entropys"])
            train_epistemics = torch.cat(self.train_metrics["epistemics"])
            train_aleatorics = torch.cat(self.train_metrics["aleatorics"])
            train_correct = torch.cat(self.train_metrics["correct"])

            wandb.log({
                "train_loss":
                    np.mean(self.train_metrics["losses"]),
                "train_complexity_cost":
                    np.mean(self.train_metrics["complexity_costs"]),
                "train_likelihood_cost":
                    np.mean(self.train_metrics["likelihood_costs"]),
                "train_accuracy":
                    (self.train_metrics["num_correct"] /
                     self.train_metrics["total"]),
                "train_time": train_time,

                "test_loss":
                    np.mean(self.test_metrics["losses"]),
                "test_complexity_cost":
                    np.mean(self.test_metrics["complexity_costs"]),
                "test_likelihood_cost":
                    np.mean(self.test_metrics["likelihood_costs"]),
                "test_accuracy":
                    (self.test_metrics["num_correct"] /
                     self.test_metrics["total"]),

                "test_entropy":
                    torch.mean(test_entropys).item(),
                "test_epistemic":
                    torch.mean(test_epistemics).item(),
                "test_aleatoric":
                    torch.mean(test_aleatorics).item(),

                "test_entropy_correct":
                    torch.mean(test_entropys[test_correct]).item(),
                "test_epistemic_correct":
                    torch.mean(test_epistemics[test_correct]).item(),
                "test_aleatoric_correct":
                    torch.mean(test_aleatorics[test_correct]).item(),

                "test_entropy_incorrect":
                    torch.mean(test_entropys[~test_correct]).item(),
                "test_epistemic_incorrect":
                    torch.mean(test_epistemics[~test_correct]).item(),
                "test_aleatoric_incorrect":
                    torch.mean(test_aleatorics[~test_correct]).item(),

                "train_entropy":
                    torch.mean(train_entropys).item(),
                "train_epistemic":
                    torch.mean(train_epistemics).item(),
                "train_aleatoric":
                    torch.mean(train_aleatorics).item(),

                "train_entropy_correct":
                    torch.mean(train_entropys[train_correct]).item(),
                "train_epistemic_correct":
                    torch.mean(train_epistemics[train_correct]).item(),
                "train_aleatoric_correct":
                    torch.mean(train_aleatorics[train_correct]).item(),

                "train_entropy_incorrect":
                    torch.mean(train_entropys[~train_correct]).item(),
                "train_epistemic_incorrect":
                    torch.mean(train_epistemics[~train_correct]).item(),
                "train_aleatoric_incorrect":
                    torch.mean(train_aleatorics[~train_correct]).item(),

                "train_dataset_size": self.train_metrics["training_size"],
                "test_time": test_time,

                "example_imgs":
                    self.get_examples_figure(self.test_metrics["examples"]),

                "epoch": epoch+1
            })

            self.reset_metrics()

    def get_examples_figure(self, examples):
        fig = make_subplots(rows=5,
                            cols=2,
                            column_widths=[0.3, 0.7])

        # annotations = []

        for i, example in enumerate(examples[:5]):
            data, target, probs = example
            img = np.squeeze(np.flip(data, axis=1))

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

            for cls, sampled_probs in enumerate(probs.T):
                fig.add_trace(
                    go.Box(y=sampled_probs,
                           name=cls,
                           showlegend=False,
                           boxmean='sd'),
                    row=i+1,
                    col=2
                )

                fig.update_yaxes(range=[-0.1, 1.1], row=i+1, col=2)

            # annotation = dict(
            #     xref="x{}".format(i*2+1),
            #     yref="y{}".format(i*2+1),
            #     showarrow=False,
            #     text="yahoo"
            # )

        return fig
