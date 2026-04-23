import torch
import torch.nn as nn
import time
import os
import numpy as np
from Radam import RAdam
from lookahead import Lookahead
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch_size, gradient_accumulation):
        self.model = model

        weight_p, bias_p = [], []
        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        
        self.optimizer_inner = RAdam(
            [{'params': weight_p, 'weight_decay': weight_decay}, 
             {'params': bias_p, 'weight_decay': 0}], 
            lr=lr
        )
        self.optimizer = Lookahead(self.optimizer_inner, la_steps=5, la_alpha=0.5)

        self.batch_size = batch_size
        self.gradient_accumulation = gradient_accumulation
        

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10, 
            T_mult=2, 
            eta_min=1e-5
        )
    
    def train(self, dataloader, device, epoch):
        self.model.train()
        Loss = nn.MSELoss()
        loss_total = 0
        self.optimizer.zero_grad() 
        
        current_count = 0
        spent_time_accumulation = 0
        len_loader = len(dataloader)

        for step, batch in enumerate(dataloader):
            start_time_batch = time.time()
            
            batch = batch.to(device, non_blocking=True)
            
            predict_labels = self.model(batch)
            
            raw_loss = Loss(predict_labels.float(), batch.y.float())
            
            if torch.isnan(raw_loss):
                print(f"\n NaN Loss detected at Epoch {epoch} Step {step}. Skipping batch.")
                self.optimizer.zero_grad()
                continue

            loss_total += raw_loss.item()
            
            scaled_loss = raw_loss / self.gradient_accumulation 

            scaled_loss.backward()
            
            if (step + 1) % self.gradient_accumulation == 0 or (step + 1) == len_loader:

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.scheduler.step(epoch + step / len_loader)

            end_time_batch = time.time()
            seconds = end_time_batch - start_time_batch
            spent_time_accumulation += seconds
            
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            spend_time_batch = "%02d:%02d:%02d" % (h, m, s)
            
            m, s = divmod(spent_time_accumulation, 60)
            h, m = divmod(m, 60)
            have_spent_time = "%02d:%02d:%02d" % (h, m, s)   

            current_count += 1
            print(f"Batch: {current_count}/{len_loader} | Time: {spend_time_batch} | Elapsed: {have_spent_time} | Loss: {raw_loss.item():.4f}", end='\r')

        print("") 
        return loss_total / (step + 1), np.array([]), np.array([])
    
    def save_model(self, model, filename):
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), filename)

class Tester(object):
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def test(self, dataloader, device, all_count):
        self.model.eval()
        Loss = nn.MSELoss() 
        loss_total = 0
        
        pred_list = []
        real_list = []
        
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                batch = batch.to(device, non_blocking=True)
                
                predict_labels = self.model(batch)
                
                loss = Loss(predict_labels.float(), batch.y.float())
                loss_total += loss.item()
                
                pred_list.append(predict_labels.cpu())
                real_list.append(batch.y.cpu())
                
        avg_loss = loss_total / len(dataloader)
        
        all_predict_labels = torch.cat(pred_list, dim=0)
        all_real_labels = torch.cat(real_list, dim=0)
        
        return avg_loss, all_real_labels.numpy().flatten(), all_predict_labels.numpy().flatten()