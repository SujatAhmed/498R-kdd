from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

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

    def _save_predictions_real_scale(self, preds, trues, data_set, flag, epoch, pred_len):
        """Save predictions in real scale using direct inverse transformation"""
        # Create directory if it doesn't exist
        pred_dir = os.path.join('./predictions/', self.args.model_id)
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        
        # Store original shapes
        preds_shape = preds.shape
        trues_shape = trues.shape
        
        # Reshape for inverse transform
        preds = preds.reshape(-1, preds_shape[-1])
        trues = trues.reshape(-1, trues_shape[-1])

        # Apply inverse transformation
        preds = data_set.inverse_transform(preds)
        trues = data_set.inverse_transform(trues)

        # Reshape back to original
        preds = preds.reshape(preds_shape)
        trues = trues.reshape(trues_shape)

        # Save the predictions and ground truth values as npy files
        if flag == 'test':
            preds_filename = os.path.join(pred_dir, f'test_pred_{pred_len}.npy')
            trues_filename = os.path.join(pred_dir, f'test_true_{pred_len}.npy')
        else:
            preds_filename = os.path.join(pred_dir, f'epoch{epoch}_{flag}_pred_{pred_len}.npy')
            trues_filename = os.path.join(pred_dir, f'epoch{epoch}_{flag}_true_{pred_len}.npy')

        np.save(preds_filename, preds)
        np.save(trues_filename, trues)
        
        print(f"Saved {flag} predictions in real scale for epoch {epoch}, pred_len {pred_len}")

    def vali(self, vali_data, vali_loader, criterion, epoch=None, save_predictions=False):
        total_loss = []
        all_preds = []
        all_trues = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if save_predictions:
                    all_preds.append(pred.numpy())
                    all_trues.append(true.numpy())

                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        
        # Save validation predictions if requested
        if save_predictions and epoch is not None and len(all_preds) > 0:
            all_preds = np.concatenate(all_preds, axis=0)
            all_trues = np.concatenate(all_trues, axis=0)
            self._save_predictions_real_scale(all_preds, all_trues, vali_data, 'val', epoch, self.args.pred_len)
        
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
            train_preds = []
            train_trues = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # Store training predictions
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                train_preds.append(pred)
                train_trues.append(true)

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
            
            # Save training predictions for this epoch in real scale
            if len(train_preds) > 0:
                train_preds = np.concatenate(train_preds, axis=0)
                train_trues = np.concatenate(train_trues, axis=0)
                self._save_predictions_real_scale(train_preds, train_trues, train_data, 'train', epoch + 1, self.args.pred_len)
            
            # Validate and save validation predictions in real scale
            vali_loss = self.vali(vali_data, vali_loader, criterion, epoch + 1, save_predictions=True)
            test_loss = self.vali(test_data, test_loader, criterion)

            train_loss = np.average(train_loss)
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
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                preds.append(outputs)
                trues.append(batch_y)
                
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if len(preds) == 0:
            print("No test predictions collected!")
            return

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # Save test predictions in real scale using your direct method
        self._save_predictions_real_scale(preds, trues, test_data, 'test', 0, self.args.pred_len)

        # Apply inverse transformation for metric calculation
        preds_shape = preds.shape
        trues_shape = trues.shape
        preds_for_metric = preds.reshape(-1, preds_shape[-1])
        trues_for_metric = trues.reshape(-1, trues_shape[-1])
        preds_real = test_data.inverse_transform(preds_for_metric).reshape(preds_shape)
        trues_real = test_data.inverse_transform(trues_for_metric).reshape(trues_shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds_real, trues_real)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds_real)
        np.save(folder_path + 'true.npy', trues_real)

        return


    
###########################  Auto-regressive test One (Adjusst stride if you need faster inference)   ##################################
    

    # def test(self, setting, test=0, auto_regressive_stride=1):
    #     test_data, test_loader = self._get_data(flag='test')
    #     if test:
    #         print('loading model')
    #         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
    
    #     preds = []
    #     trues = []
    #     folder_path = './test_results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    
    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)
    
    #             if 'PEMS' in self.args.data or 'Solar' in self.args.data:
    #                 batch_x_mark = None
    #                 batch_y_mark = None
    #                 use_marks = False
    #             else:
    #                 batch_x_mark = batch_x_mark.float().to(self.device)
    #                 batch_y_mark = batch_y_mark.float().to(self.device)
    #                 use_marks = True
    
    #             seq_len = self.args.seq_len
    #             label_len = self.args.label_len
    #             total_pred_len = self.args.pred_len
    #             stride = auto_regressive_stride
    
    #             # Build full time marks for the entire period [0, seq_len+pred_len-1]
    #             if use_marks:
    #                 full_marks = torch.zeros(1, seq_len + total_pred_len, batch_x_mark.shape[-1]).to(self.device)
    #                 full_marks[:, :seq_len, :] = batch_x_mark
    #                 # Fill future marks from batch_y_mark (which starts at index seq_len - label_len)
    #                 for t in range(seq_len, seq_len + total_pred_len):
    #                     y_idx = t - (seq_len - label_len)
    #                     if y_idx < batch_y_mark.shape[1]:
    #                         full_marks[:, t, :] = batch_y_mark[:, y_idx, :]
    #                     else:
    #                         full_marks[:, t, :] = batch_y_mark[:, -1, :]   # fallback (should not happen)
    #             else:
    #                 full_marks = None
    
    #             current_seq = batch_x.clone()   # (1, seq_len, F)
    #             full_pred = torch.zeros(1, total_pred_len, batch_x.shape[-1]).to(self.device)
    
    #             pos = 0
    #             while pos < total_pred_len:
    #                 # Encoder input: last seq_len values of extended series
    #                 if pos == 0:
    #                     enc_input = current_seq
    #                 else:
    #                     extended = torch.cat([current_seq, full_pred[:, :pos, :]], dim=1)
    #                     enc_input = extended[:, -seq_len:, :]
    
    #                 # Decoder input: last label_len values of extended series
    #                 if pos == 0:
    #                     known_vals = current_seq[:, -label_len:, :]
    #                 else:
    #                     extended = torch.cat([current_seq, full_pred[:, :pos, :]], dim=1)
    #                     if pos >= label_len:
    #                         known_vals = extended[:, -label_len:, :]
    #                     else:
    #                         known_vals = torch.cat([current_seq[:, -(label_len - pos):, :],
    #                                                 full_pred[:, :pos, :]], dim=1)
    #                 dec_inp = torch.zeros(1, label_len + total_pred_len, batch_x.shape[-1]).to(self.device)
    #                 dec_inp[:, :label_len, :] = known_vals
    
    #                 # Time marks for this step
    #                 if use_marks:
    #                     enc_mark = full_marks[:, pos:pos+seq_len, :]
    #                     dec_mark = full_marks[:, pos:pos+label_len+total_pred_len, :]
    #                 else:
    #                     enc_mark = None
    #                     dec_mark = None
    
    #                 # Forward pass
    #                 if self.args.use_amp:
    #                     with torch.cuda.amp.autocast():
    #                         if self.args.output_attention:
    #                             outputs = self.model(enc_input, enc_mark, dec_inp, dec_mark)[0]
    #                         else:
    #                             outputs = self.model(enc_input, enc_mark, dec_inp, dec_mark)
    #                 else:
    #                     if self.args.output_attention:
    #                         outputs = self.model(enc_input, enc_mark, dec_inp, dec_mark)[0]
    #                     else:
    #                         outputs = self.model(enc_input, enc_mark, dec_inp, dec_mark)
    
    #                 f_dim = -1 if self.args.features == 'MS' else 0
    #                 outputs = outputs[:, -total_pred_len:, f_dim:]
    #                 remaining = total_pred_len - pos
    #                 take = min(stride, remaining)
    #                 full_pred[:, pos:pos+take, :] = outputs[:, :take, :]
    #                 pos += take
    
    #             outputs = full_pred.detach().cpu().numpy()
    #             batch_y_true = batch_y[:, -total_pred_len:, f_dim:].detach().cpu().numpy()
    
    #             preds.append(outputs)
    #             trues.append(batch_y_true)
    
    #             # Optional visualisation (unchanged)
    #             if i % 20 == 0:
    #                 input_np = batch_x.detach().cpu().numpy()
    #                 if test_data.scale and self.args.inverse:
    #                     shape = input_np.shape
    #                     input_np = test_data.inverse_transform(input_np.squeeze(0)).reshape(shape)
    #                 gt = np.concatenate((input_np[0, :, -1], batch_y_true[0, :, -1]), axis=0)
    #                 pd = np.concatenate((input_np[0, :, -1], outputs[0, :, -1]), axis=0)
    #                 visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
    
    #     if len(preds) == 0:
    #         print("No test predictions collected!")
    #         return
    
    #     preds = np.concatenate(preds, axis=0)
    #     trues = np.concatenate(trues, axis=0)
    #     print('test shape:', preds.shape, trues.shape)
    
    #     # Save raw predictions (original scale) – unchanged
    #     self._save_predictions_real_scale(preds, trues, test_data, 'test', 0, self.args.pred_len)
    
    #     # Inverse transform for metric calculation – unchanged
    #     preds_shape = preds.shape
    #     trues_shape = trues.shape
    #     preds_flat = preds.reshape(-1, preds_shape[-1])
    #     trues_flat = trues.reshape(-1, trues_shape[-1])
    #     preds_real = test_data.inverse_transform(preds_flat).reshape(preds_shape)
    #     trues_real = test_data.inverse_transform(trues_flat).reshape(trues_shape)
    
    #     # Save metrics – unchanged
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    
    #     mae, mse, rmse, mape, mspe = metric(preds_real, trues_real)
    #     print('mse:{}, mae:{}'.format(mse, mae))
    #     with open("result_long_term_forecast.txt", 'a') as f:
    #         f.write(setting + "  \n")
    #         f.write('mse:{}, mae:{}'.format(mse, mae))
    #         f.write('\n\n')
    
    #     np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    #     np.save(folder_path + 'pred.npy', preds_real)
    #     np.save(folder_path + 'true.npy', trues_real)
    
    #     return


    
# ############################################  Auto-regressive test two   #######################################


    # def test(self, setting, test=0):
    #     test_data, test_loader = self._get_data(flag='test')
    #     if test:
    #         print('loading model')
    #         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
    
    #     preds = []
    #     trues = []
    #     folder_path = './test_results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    
    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)
    
    #             if 'PEMS' in self.args.data or 'Solar' in self.args.data:
    #                 batch_x_mark = None
    #                 batch_y_mark = None
    #             else:
    #                 batch_x_mark = batch_x_mark.float().to(self.device)
    #                 batch_y_mark = batch_y_mark.float().to(self.device)
    
    #             # ------------------------------------------------------------------ #
    #             # AUTO-REGRESSIVE INFERENCE                                          #
    #             # The encoder always sees a real or previously-predicted 96-step     #
    #             # window. Future ground-truth values are never consumed.             #
    #             # ------------------------------------------------------------------ #
    #             seq_len   = self.args.seq_len    # 96 (fixed)
    #             label_len = self.args.label_len
    #             pred_len  = self.args.pred_len   # 96 / 192 / 336 / 720
    #             f_dim     = -1 if self.args.features == 'MS' else 0
    #             B, _, C   = batch_x.shape
    
    #             # Running encoder buffer – starts with real observations
    #             enc_buffer = batch_x.clone()     # [B, seq_len, C]
    
    #             # Pre-build a full time-mark strip for slicing later
    #             use_marks = batch_x_mark is not None
    #             if use_marks:
    #                 full_mark = torch.cat([batch_x_mark, batch_y_mark], dim=1)
    #                 enc_mark_buffer = batch_x_mark.clone()
    
    #             collected_preds = []
    #             steps_needed = int(np.ceil(pred_len / seq_len))
    
    #             for step in range(steps_needed):
    #                 # -- encoder marks --
    #                 cur_enc_mark = enc_mark_buffer if use_marks else None
    
    #                 # -- decoder input: label from enc tail + zeros placeholder --
    #                 dec_label = enc_buffer[:, -label_len:, :]          # [B, label_len, C]
    #                 dec_zeros = torch.zeros(B, seq_len, C, device=self.device)
    #                 dec_inp   = torch.cat([dec_label, dec_zeros], dim=1).float()
    
    #                 # -- decoder marks --
    #                 if use_marks:
    #                     enc_start   = step * seq_len
    #                     dm_start    = enc_start + seq_len - label_len
    #                     dm_end      = dm_start + label_len + seq_len
    #                     cur_dec_mark = full_mark[:, dm_start:dm_end, :]
    #                 else:
    #                     cur_dec_mark = None
    
    #                 # -- forward pass --
    #                 if self.args.use_amp:
    #                     with torch.cuda.amp.autocast():
    #                         if self.args.output_attention:
    #                             out = self.model(enc_buffer, cur_enc_mark, dec_inp, cur_dec_mark)[0]
    #                         else:
    #                             out = self.model(enc_buffer, cur_enc_mark, dec_inp, cur_dec_mark)
    #                 else:
    #                     if self.args.output_attention:
    #                         out = self.model(enc_buffer, cur_enc_mark, dec_inp, cur_dec_mark)[0]
    #                     else:
    #                         out = self.model(enc_buffer, cur_enc_mark, dec_inp, cur_dec_mark)
    
    #                 chunk = out[:, -seq_len:, f_dim:]                  # [B, 96, f_dim]
    #                 collected_preds.append(chunk)
    
    #                 # -- slide encoder buffer forward by 96 steps --
    #                 if f_dim == -1 and C > 1:                          # MS mode: pad back to C
    #                     pad        = torch.zeros(B, seq_len, C - 1, device=self.device)
    #                     chunk_full = torch.cat([pad, chunk], dim=-1)
    #                 else:
    #                     chunk_full = chunk
    #                 enc_buffer = torch.cat([enc_buffer[:, seq_len:, :], chunk_full], dim=1)
    
    #                 # -- slide encoder mark buffer --
    #                 if use_marks:
    #                     nm_start        = (step + 1) * seq_len
    #                     nm_end          = nm_start + seq_len
    #                     enc_mark_buffer = full_mark[:, nm_start:nm_end, :]
    
    #             # Trim to exact pred_len
    #             outputs = torch.cat(collected_preds, dim=1)[:, :pred_len, :]  # [B, pred_len, f_dim]
    
    #             # ------------------------------------------------------------------ #
    #             # Everything below is identical to the original test function        #
    #             # ------------------------------------------------------------------ #
    #             f_dim   = -1 if self.args.features == 'MS' else 0
    #             outputs = outputs.detach().cpu().numpy()
    #             batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
    #             batch_y = batch_y.detach().cpu().numpy()
    
    #             if test_data.scale and self.args.inverse:
    #                 shape   = outputs.shape
    #                 outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
    #                 batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
    
    #             pred = outputs
    #             true = batch_y
    
    #             preds.append(pred)
    #             trues.append(true)
    #             if i % 20 == 0:
    #                 input = batch_x.detach().cpu().numpy()
    #                 if test_data.scale and self.args.inverse:
    #                     shape = input.shape
    #                     input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
    #                 gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
    #                 pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
    #                 visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
    
    #     preds = np.array(preds)
    #     trues = np.array(trues)
    #     print('test shape:', preds.shape, trues.shape)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    #     trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    #     print('test shape:', preds.shape, trues.shape)
    
    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    
    #     # Save predictions in real scale
    #     self._save_predictions_real_scale(preds, trues, test_data, 'test', 0, self.args.pred_len)

    #     # Inverse transform for metric calculation
    #     preds_shape = preds.shape
    #     trues_shape = trues.shape
    #     preds_real = test_data.inverse_transform(preds.reshape(-1, preds_shape[-1])).reshape(preds_shape)
    #     trues_real = test_data.inverse_transform(trues.reshape(-1, trues_shape[-1])).reshape(trues_shape)

    #     mae, mse, rmse, mape, mspe = metric(preds_real, trues_real)
    #     print('mse:{}, mae:{}'.format(mse, mae))
    #     f = open("result_long_term_forecast.txt", 'a')
    #     f.write(setting + "  \n")
    #     f.write('mse:{}, mae:{}'.format(mse, mae))
    #     f.write('\n')
    #     f.write('\n')
    #     f.close()

    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    #     np.save(folder_path + 'pred.npy', preds_real)
    #     np.save(folder_path + 'true.npy', trues_real)
    
    #     return



    
# -----------------------------------------------------------------------------------------
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
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
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return