from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _split_timesnetrange_outputs(self, outputs):
        """Extract mean/lower/upper tensors from TimesNetRange output.

        ``TimesNetRange`` may return either a tuple of three tensors or a
        single stacked tensor whose first or second dimension indexes the
        statistics.  This helper normalises the output format so the rest of
        the training/validation/test code can assume three separate tensors.
        """

        if isinstance(outputs, (list, tuple)):
            # Already a sequence of tensors – return the first three values.
            return outputs[0], outputs[1], outputs[2]

        # When ``outputs`` is a tensor we expect the statistics dimension to
        # be either the first or second axis.  Handle both layouts.
        if outputs.ndim > 1 and outputs.shape[0] == 3:
            return outputs[0], outputs[1], outputs[2]

        return outputs[:, 0], outputs[:, 1], outputs[:, 2]
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if self.args.model == 'TimesNetRange':
                    mean_pred, lower_pred, upper_pred = self._split_timesnetrange_outputs(outputs)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    y_mean = batch_y.mean(dim=1)
                    y_lower = batch_y.min(dim=1).values
                    y_upper = batch_y.max(dim=1).values
                    loss = (
                        criterion(mean_pred, y_mean)
                        + criterion(lower_pred, y_lower)
                        + criterion(upper_pred, y_upper)
                    )
                    loss = loss.detach().cpu()
                    pred = mean_pred.detach().cpu()
                    true = y_mean.detach().cpu()
                else:
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()
                    loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if self.args.model == 'TimesNetRange':
                    mean_pred, lower_pred, upper_pred = self._split_timesnetrange_outputs(outputs)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    y_mean = batch_y.mean(dim=1)
                    y_lower = batch_y.min(dim=1).values
                    y_upper = batch_y.max(dim=1).values
                    loss = (
                        criterion(mean_pred, y_mean)
                        + criterion(lower_pred, y_lower)
                        + criterion(upper_pred, y_upper)
                    )
                else:
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        # store statistics separately when model outputs range predictions
        range_mean_preds, range_mean_trues = [], []
        range_min_preds, range_min_trues = [], []
        range_max_preds, range_max_trues = [], []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if self.args.model == 'TimesNetRange':
                    mean_pred, lower_pred, upper_pred = self._split_timesnetrange_outputs(outputs)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    y_mean = batch_y[:, :, f_dim:].mean(dim=1)
                    y_lower = batch_y[:, :, f_dim:].min(dim=1).values
                    y_upper = batch_y[:, :, f_dim:].max(dim=1).values

                    # convert to numpy
                    mean_pred = mean_pred.detach().cpu().numpy()
                    lower_pred = lower_pred.detach().cpu().numpy()
                    upper_pred = upper_pred.detach().cpu().numpy()
                    y_mean = y_mean.detach().cpu().numpy()
                    y_lower = y_lower.detach().cpu().numpy()
                    y_upper = y_upper.detach().cpu().numpy()

                    if test_data.scale and self.args.inverse:
                        mean_pred = test_data.inverse_transform(mean_pred)
                        lower_pred = test_data.inverse_transform(lower_pred)
                        upper_pred = test_data.inverse_transform(upper_pred)
                        y_mean = test_data.inverse_transform(y_mean)
                        y_lower = test_data.inverse_transform(y_lower)
                        y_upper = test_data.inverse_transform(y_upper)

                    # store mean/min/max for later evaluation
                    range_mean_preds.append(mean_pred)
                    range_mean_trues.append(y_mean)
                    range_min_preds.append(lower_pred)
                    range_min_trues.append(y_lower)
                    range_max_preds.append(upper_pred)
                    range_max_trues.append(y_upper)

                    pred = mean_pred
                    true = y_mean
                else:
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = batch_y.shape
                        if outputs.shape[-1] != batch_y.shape[-1]:
                            outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                        outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:]

                    pred = outputs
                    true = batch_y

                if self.args.model != 'TimesNetRange':
                    preds.append(pred)
                    trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)

                    if input.ndim == 3 and true.ndim == 3 and pred.ndim == 3:
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    else:
                        # When prediction/true tensors are reduced (e.g. TimesNetRange)
                        # their dimensionality may not match ``input``.  Flatten all
                        # arrays to 1D before concatenation to avoid shape errors.
                        last_input = input[:, -1] if input.ndim > 1 else input
                        last_true = true[:, -1] if true.ndim > 1 else true
                        last_pred = pred[:, -1] if pred.ndim > 1 else pred
                        gt = np.concatenate((last_input.reshape(-1), last_true.reshape(-1)), axis=0)
                        pd = np.concatenate((last_input.reshape(-1), last_pred.reshape(-1)), axis=0)

                    # ``TimesNetRange`` produces aggregate statistics rather than
                    # full sequences.  Visualizing them alongside the raw input is
                    # not meaningful and previously caused a crash.  Skip plotting
                    # when using this model.
                    if self.args.model != 'TimesNetRange':
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.model == 'TimesNetRange':
            mean_preds = np.concatenate(range_mean_preds, axis=0)
            mean_trues = np.concatenate(range_mean_trues, axis=0)
            min_preds = np.concatenate(range_min_preds, axis=0)
            min_trues = np.concatenate(range_min_trues, axis=0)
            max_preds = np.concatenate(range_max_preds, axis=0)
            max_trues = np.concatenate(range_max_trues, axis=0)

            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            mean_metrics = metric(mean_preds, mean_trues)
            min_metrics = metric(min_preds, min_trues)
            max_metrics = metric(max_preds, max_trues)

            print('mean mse:{}, mae:{}'.format(mean_metrics[1], mean_metrics[0]))
            print('min mse:{}, mae:{}'.format(min_metrics[1], min_metrics[0]))
            print('max mse:{}, mae:{}'.format(max_metrics[1], max_metrics[0]))

            with open("result_long_term_forecast.txt", 'a') as f:
                f.write(setting + "  \n")
                f.write('mean mse:{}, mae:{}\n'.format(mean_metrics[1], mean_metrics[0]))
                f.write('min mse:{}, mae:{}\n'.format(min_metrics[1], min_metrics[0]))
                f.write('max mse:{}, mae:{}\n'.format(max_metrics[1], max_metrics[0]))
                f.write('\n')

            np.save(folder_path + 'mean_metrics.npy', np.array(mean_metrics))
            np.save(folder_path + 'min_metrics.npy', np.array(min_metrics))
            np.save(folder_path + 'max_metrics.npy', np.array(max_metrics))
            np.save(folder_path + 'pred_mean.npy', mean_preds)
            np.save(folder_path + 'true_mean.npy', mean_trues)
            np.save(folder_path + 'pred_min.npy', min_preds)
            np.save(folder_path + 'true_min.npy', min_trues)
            np.save(folder_path + 'pred_max.npy', max_preds)
            np.save(folder_path + 'true_max.npy', max_trues)

            return
        else:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            print('test shape:', preds.shape, trues.shape)
            if preds.ndim == 3:
                preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
                trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
                print('test shape:', preds.shape, trues.shape)
            else:
                preds = preds.reshape(-1, preds.shape[-1])
                trues = trues.reshape(-1, trues.shape[-1])
                print('aggregated shape:', preds.shape, trues.shape)

            # result save
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # dtw calculation only for sequence outputs
            if self.args.use_dtw and preds.ndim == 3:
                dtw_list = []
                manhattan_distance = lambda x, y: np.abs(x - y)
                for i in range(preds.shape[0]):
                    x = preds[i].reshape(-1, 1)
                    y = trues[i].reshape(-1, 1)
                    if i % 100 == 0:
                        print("calculating dtw iter:", i)
                    d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                    dtw_list.append(d)
                dtw = np.array(dtw_list).mean()
            else:
                dtw = 'Not calculated'

            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
            f = open("result_long_term_forecast.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
            f.write('\n')
            f.write('\n')
            f.close()

            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

            return
