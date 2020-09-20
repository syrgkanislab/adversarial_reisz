import os
import numpy as np
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from .oadam import OAdam


def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


class DeepReisz:

    def __init__(self, learner, adversary, moment_fn):
        """
        Parameters
        ----------
        learner : a pytorch neural net module
        adversary : a pytorch neural net module
        moment_fn : a function that takes as input a tuple (x, learner, adversary) and
            evaluates the moment function at teach of the x's, with models learner, adversary.
            The learner and adversary are torch models that take as input x and return the
            outcome of each model.
        """
        self.learner = learner
        self.adversary = adversary
        self.moment_fn = moment_fn

    def _pretrain(self, X,
                  learner_l2, adversary_l2, adversary_norm_reg,
                  learner_lr, adversary_lr, n_epochs, bs, train_learner_every, train_adversary_every,
                  warm_start, logger, model_dir, device, verbose):
        """ Prepares the variables required to begin training.
        """
        self.verbose = verbose

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.tempdir = tempfile.TemporaryDirectory(dir=model_dir)
        self.model_dir = self.tempdir.name

        self.n_epochs = n_epochs
        self.train_ds = TensorDataset(X)
        self.train_dl = DataLoader(self.train_ds, batch_size=bs, shuffle=True)

        self.learner = self.learner.to(device)
        self.adversary = self.adversary.to(device)

        if not warm_start:
            self.learner.apply(lambda m: (
                m.reset_parameters() if hasattr(m, 'reset_parameters') else None))
            self.adversary.apply(lambda m: (
                m.reset_parameters() if hasattr(m, 'reset_parameters') else None))

        beta1 = 0.
        self.optimizerD = OAdam(add_weight_decay(self.learner, learner_l2),
                                lr=learner_lr, betas=(beta1, .01))
        self.optimizerG = OAdam(add_weight_decay(
            self.adversary, adversary_l2), lr=adversary_lr, betas=(beta1, .01))

        if logger is not None:
            self.writer = SummaryWriter()

        return X

    def fit(self, X,
            learner_l2=1e-3, adversary_l2=1e-4, adversary_norm_reg=1e-3,
            learner_lr=0.001, adversary_lr=0.001, n_epochs=100, bs=100, train_learner_every=1, train_adversary_every=1,
            warm_start=False, logger=None, model_dir='.', device=None, verbose=0):
        """
        Parameters
        ----------
        X : features
        learner_l2, adversary_l2 : l2_regularization of parameters of learner and adversary
        adversary_norm_reg : adveresary norm regularization weight
        learner_lr : learning rate of the Adam optimizer for learner
        adversary_lr : learning rate of the Adam optimizer for adversary
        n_epochs : how many passes over the data
        bs : batch size
        train_learner_every : after how many training iterations of the adversary should we train the learner
        warm_start : if False then network parameters are initialized at the beginning, otherwise we start
            from their current weights
        logger : a function that takes as input (learner, adversary, epoch, writer) and is called after every epoch
            Supposed to be used to log the state of the learning.
        model_dir : folder where to store the learned models after every epoch
        """

        X = self._pretrain(X,
                           learner_l2, adversary_l2, adversary_norm_reg,
                           learner_lr, adversary_lr, n_epochs, bs, train_learner_every, train_adversary_every,
                           warm_start, logger, model_dir, device, verbose)

        for epoch in range(n_epochs):

            if self.verbose > 0:
                print("Epoch #", epoch, sep="")

            for it, (xb,) in enumerate(self.train_dl):
                xb = xb.to(device)

                if (it % train_learner_every == 0):
                    self.learner.train()
                    D_loss = torch.mean(self.moment_fn(
                        xb, self.learner, self.adversary))
                    self.optimizerD.zero_grad()
                    D_loss.backward()
                    self.optimizerD.step()
                    self.learner.eval()

                if (it % train_adversary_every == 0):
                    self.adversary.train()
                    test = self.adversary(xb)
                    G_loss = - torch.mean(self.moment_fn(
                        xb, self.learner, self.adversary)) + torch.mean(test**2)
                    self.optimizerG.zero_grad()
                    G_loss.backward()
                    self.optimizerG.step()
                    self.adversary.eval()

            torch.save(self.learner, os.path.join(
                self.model_dir, "epoch{}".format(epoch)))

            if logger is not None:
                logger(self.learner, self.adversary, epoch, self.writer)

        if logger is not None:
            self.writer.flush()
            self.writer.close()

        return self

    def predict(self, T, model='avg', burn_in=0, alpha=None):
        """
        Parameters
        ----------
        X : (n, p) matrix of features
        model : one of ('avg', 'final'), whether to use an average of models or the final
        burn_in : discard the first "burn_in" epochs when doing averaging
        alpha : if not None but a float, then it also returns the a/2 and 1-a/2, percentile of
            the predictions across different epochs (proxy for a confidence interval)
        Returns
        -------
        a : (n,) vector of learned reisz representers a(X)
        """
        if model == 'avg':
            preds = np.array([torch.load(os.path.join(self.model_dir,
                                                      "epoch{}".format(i)))(T).cpu().data.numpy()
                              for i in np.arange(burn_in, self.n_epochs)])
            if alpha is None:
                return np.mean(preds, axis=0).flatten()
            else:
                return np.mean(preds, axis=0).flatten(),\
                    np.percentile(preds, 100 * alpha / 2, axis=0).flatten(),\
                    np.percentile(preds, 100 * (1 - alpha / 2),
                                  axis=0).flatten()
        if model == 'final':
            return torch.load(os.path.join(self.model_dir,
                                           "epoch{}".format(self.n_epochs - 1)))(T).cpu().data.numpy().flatten()
        if isinstance(model, int):
            return torch.load(os.path.join(self.model_dir,
                                           "epoch{}".format(model)))(T).cpu().data.numpy().flatten()
