import torch as t
import numpy as np
import os
from Params import args
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)
from sklearn.preprocessing import label_binarize
import copy
import random
from Utils import contrastLoss, keras_style_init, calcRegLoss
from Model import Model
from DataHandler import DataHandler
import TimeLogger as logger
from TimeLogger import log


 
class EnhancedClassifierLayer(t.nn.Module):
    def __init__(self, initializer='glorot_uniform'):
        super(EnhancedClassifierLayer, self).__init__()
        self.lin1 = t.nn.Linear(args.latdim * 2, 256)
        self.lin2 = t.nn.Linear(256, 128)
        self.lin3 = t.nn.Linear(128, args.num_classes)
        self.dropout = t.nn.Dropout(0.5)
        self.bn1 = t.nn.BatchNorm1d(256)
        self.bn2 = t.nn.BatchNorm1d(128)

        # Apply Keras-style initialization to weights
        keras_style_init(self.lin1.weight, initializer)
        keras_style_init(self.lin1.bias, 'zeros')
        keras_style_init(self.lin2.weight, initializer)
        keras_style_init(self.lin2.bias, 'zeros')
        keras_style_init(self.lin3.weight, initializer)
        keras_style_init(self.lin3.bias, 'zeros')

    def forward(self, dEmbeds, gEmbeds):
        embeds = t.concat((dEmbeds, gEmbeds), 1)
        embeds = F.relu(self.bn1(self.lin1(embeds)))
        embeds = self.dropout(embeds)
        embeds = F.relu(self.bn2(self.lin2(embeds)))
        embeds = self.dropout(embeds)
        ret = self.lin3(embeds)
        return ret


class EnsembleCoach:
    def __init__(self, handler, num_models=5, model_types=None, seed_start=args.seed):
        self.handler = handler
        self.num_models = num_models
        self.model_types = model_types or ['base'] * num_models
        self.seed_start = seed_start
        self.models = []
        self.seeds = list(range(seed_start, seed_start + num_models))

        print('DRUG', args.drug, 'GENE', args.gene)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        print(f'Training ensemble of {num_models} models')

        self.metrics = dict()
        for met in ['Loss', 'preLoss', 'Acc']:
            self.metrics['Train' + met] = []
            self.metrics['Val' + met] = []
        self.metric_keys = ['acc', 'balanced_acc', 'macro_f1', 'auroc', 'aupr', 'mcc']

    def makePrint(self, name, ep, reses, save):
        ret = f"Epoch {ep}/{args.epoch}, {name}: "
        for metric in reses:
            val = reses[metric]
            ret += f"{metric} = {val:.4f}, "
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        return ret[:-2] + '  '

    def prepareModels(self, initializer='glorot_uniform'):
        for i in range(self.num_models):
            t.manual_seed(self.seeds[i])
            np.random.seed(self.seeds[i])
            random.seed(self.seeds[i])


            model = Model(initializer)

            if args.device.startswith('cuda'):
                model = model.cuda()

            optimizer = t.optim.AdamW(model.parameters(), lr=args.lr)
            scheduler = t.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=args.lr, max_lr=args.lr * 10,
                step_size_up=100, cycle_momentum=False
            )

            self.models.append({
                'model': model,
                'optimizer': optimizer,
                'scheduler': scheduler,
                'seed': self.seeds[i],
                'type': self.model_types[i]
            })

    def trainEnsemble(self):
        print('Models Prepared')

        for ep in range(args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)

            for i, model_dict in enumerate(self.models):
                self.trainModelEpoch(model_dict, ep)

                if tstFlag:
                    self.testModelEpoch(model_dict)

            if tstFlag:
                reses = self.testEnsemble()
                print(self.makePrint('Eval Ensemble', ep, reses, tstFlag))
                val_log = ', '.join([f'val_{key}={reses[f"val_{key}"]:.4f}' for key in self.metric_keys])
                test_log = ', '.join([f'test_{key}={reses[f"test_{key}"]:.4f}' for key in self.metric_keys])
                print(f'[Logging] Epoch {ep}: {val_log}; {test_log}')

        reses = self.testEnsemble()
        print(self.makePrint('Final Eval Ensemble', args.epoch, reses, True))
        self.saveEnsemble()
        return reses

    def trainModelEpoch(self, model_dict, epoch):
        model = model_dict['model']
        optimizer = model_dict['optimizer']
        scheduler = model_dict['scheduler']

        model.train()
        trnLoader = self.handler.trnLoader
        epLoss, epPreLoss = 0, 0
        steps = len(trnLoader.dataset) // args.batch

        for tem in trnLoader:
            drugs, genes, labels = tem
            labels = labels.long()
            if args.device.startswith('cuda'):
                drugs, genes, labels = drugs.cuda(), genes.cuda(), labels.cuda()
            ceLoss, sslLoss = model.calcLosses(drugs, genes, labels, self.handler.torchBiAdj, args.keepRate)
            sslLoss *= args.ssl_reg

            regLoss = calcRegLoss(model) * args.reg
            loss = ceLoss + regLoss + sslLoss

            epLoss += loss.item()
            epPreLoss += ceLoss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return {'Loss': epLoss / steps, 'preLoss': epPreLoss / steps}

    def testModelEpoch(self, model_dict):
        model = model_dict['model']
        valLoader = self.handler.valLoader
        tstLoader = self.handler.tstLoader
        model.eval()

        val_metrics = self.evaluate(model, valLoader)
        test_metrics = self.evaluate(model, tstLoader)

        res = {}
        for key in self.metric_keys:
            res[f'val_{key}'] = val_metrics.get(key, np.nan)
        for key in self.metric_keys:
            res[f'test_{key}'] = test_metrics.get(key, np.nan)
        return res

    def compute_metrics(self, labels, probs):
        labels = np.asarray(labels)
        probs = np.asarray(probs)
        preds = probs.argmax(axis=1)

        metrics = {}
        metrics['acc'] = accuracy_score(labels, preds)
        metrics['macro_f1'] = f1_score(labels, preds, average='macro', zero_division=0)
        try:
            metrics['balanced_acc'] = balanced_accuracy_score(labels, preds)
        except ValueError:
            metrics['balanced_acc'] = np.nan
        try:
            metrics['mcc'] = matthews_corrcoef(labels, preds)
        except ValueError:
            metrics['mcc'] = np.nan

        metrics['auroc'] = np.nan
        metrics['aupr'] = np.nan

        unique_labels = np.unique(labels)
        num_classes = probs.shape[1] if probs.ndim == 2 else args.num_classes
        if len(unique_labels) > 1 and probs.size > 0:
            if num_classes == 2:
                y_score = probs[:, 1]
                try:
                    metrics['auroc'] = roc_auc_score(labels, y_score)
                except ValueError:
                    metrics['auroc'] = np.nan
                try:
                    metrics['aupr'] = average_precision_score(labels, y_score)
                except ValueError:
                    metrics['aupr'] = np.nan
            else:
                y_true_bin = label_binarize(labels, classes=list(range(num_classes)))
                try:
                    metrics['auroc'] = roc_auc_score(y_true_bin, probs, multi_class='ovr', average='macro')
                except ValueError:
                    metrics['auroc'] = np.nan
                try:
                    metrics['aupr'] = average_precision_score(y_true_bin, probs, average='macro')
                except ValueError:
                    metrics['aupr'] = np.nan

        return metrics

    def evaluate(self, model, loader):
        all_probs, all_labels = [], []
        with t.no_grad():
            for drugs, genes, labels in loader:
                if args.device.startswith('cuda'):
                    drugs, genes, labels = drugs.cuda(), genes.cuda(), labels.long().cuda()
                else:
                    labels = labels.long()
                logits = model.predict(self.handler.torchBiAdj, drugs, genes)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())

        if not all_probs:
            return {key: np.nan for key in self.metric_keys}

        probs = np.concatenate(all_probs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        return self.compute_metrics(labels, probs)

    def ensemblePredict(self, loader):
        all_probs, all_labels = [], []
        for drugs, genes, labels in loader:
            if args.device.startswith('cuda'):
                drugs, genes = drugs.cuda(), genes.cuda()
            batch_probs = []
            for model_dict in self.models:
                model = model_dict['model']
                model.eval()
                with t.no_grad():
                    logits = model.predict(self.handler.torchBiAdj, drugs, genes)
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                    batch_probs.append(probs)
            mean_probs = np.mean(batch_probs, axis=0)
            all_probs.append(mean_probs)
            all_labels.append(labels.cpu().numpy())
        probs = np.concatenate(all_probs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        return probs, labels

    def testEnsemble(self):
        val_probs, val_labels = self.ensemblePredict(self.handler.valLoader)
        test_probs, test_labels = self.ensemblePredict(self.handler.tstLoader)

        val_metrics = self.compute_metrics(val_labels, val_probs)
        test_metrics = self.compute_metrics(test_labels, test_probs)

        res = {}
        for key in self.metric_keys:
            res[f'val_{key}'] = val_metrics.get(key, np.nan)
        for key in self.metric_keys:
            res[f'test_{key}'] = test_metrics.get(key, np.nan)
        return res

    def saveEnsemble(self):
        model_parent_path = os.path.join('saved_models', 'ensemble_models')
        os.makedirs(model_parent_path, exist_ok=True)
        for i, model_dict in enumerate(self.models):
            model = model_dict['model']
            model_type = model_dict['type']
            seed = model_dict['seed']
            t.save(model.state_dict(), f'{model_parent_path}/model_{i + 1}_{model_type}_seed{seed}.pkl')
        print(f'Ensemble models saved to {model_parent_path}')


if __name__ == '__main__':
    args.is_debug = True
    args.epoch = 500
    args.seed = 42
    args.latdim = 1024
    args.lr = 0.001
    args.batch = 1024 * 4
    args.data = "drugbank"
    print(args)
    
    use_cuda = args.gpu >= 0 and t.cuda.is_available()
    device = f'cuda:{args.gpu}' if use_cuda else 'cpu'
    if use_cuda:
        t.cuda.set_device(device)
    args.device = device

    logger.saveDefault = True
    log('Start Ensemble Training')

    handler = DataHandler()
    handler.LoadData()
    log('Data Loaded')

    model_types = ['base', 'variant1', 'variant2', 'variant3', 'base']
    ensemble_coach = EnsembleCoach(handler, num_models=5, model_types=model_types)

    # Specify Keras-style initializer
    # Options: 'glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform',
    #          'zeros', 'ones', 'random_normal', 'random_uniform'
    ensemble_coach.prepareModels(initializer='he_normal')
    final_metrics = ensemble_coach.trainEnsemble()

    metric_keys = getattr(ensemble_coach, 'metric_keys', ['acc', 'balanced_acc', 'macro_f1', 'auroc', 'aupr', 'mcc'])
    for split in ['val', 'test']:
        for key in metric_keys:
            metric_value = final_metrics.get(f'{split}_{key}')
            if metric_value is not None:
                metric_name = f'{split.title()} {key.replace("_", " ").title()}'
                log(f'Final Ensemble {metric_name}: {metric_value:.4f}')
