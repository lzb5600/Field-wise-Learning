import itertools
import random
import torch
import tqdm
import copy
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from gen_dataset.avazu import AvazuDataset
from gen_dataset.criteo import CriteoDataset
from fwl import field_wise_learning_model, variance_reg


def get_dataset(path):
    if   'criteo' in path:
        return CriteoDataset(path)
    elif 'avazu' in path:
        return AvazuDataset(path)
    else:
        raise ValueError('unknown dataset name')


def train(model, optimizer, data_loader, criterion, device, norm_reg, reg_freq):
    cnt=0
    model.train()
    for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step() 
        cnt+=1
        if cnt%reg_freq == 0:
            norm_reg.step(model)
    

def test(model, data_loader, criterion, device):
    model.eval()
    targets, predicts = list(), list()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for fields, target in data_loader:
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            loss = criterion(y, target.float())
            total_loss += loss.item()*len(y)
            total_samples += len(y)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts),total_loss/total_samples


def main(args):
    device = torch.device(args.device)
    dataset = get_dataset(args.dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    all_index = list(range(len(dataset)))      
    random.seed(6)
    random.shuffle(all_index)
    train_idx = all_index[:train_length]
    val_idx = all_index[train_length:(train_length + valid_length)]
    test_idx = all_index[(train_length + valid_length):]
    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size)    
    model = field_wise_learning_model(dataset.field_dims, args.ebd_dim, log_ebd=args.log_ebd, include_linear=args.include_linear).to(device)       
    norm_reg = variance_reg(num_fields=len(dataset.field_dims), reg_lr=args.reg_lr, reg_mean=args.reg_mean, reg_adagrad=args.reg_adagrad, reg_sqrt=args.reg_sqrt)
    criterion = torch.nn.BCEWithLogitsLoss()    
    optimizer = torch.optim.Adagrad([{'params': model.feature_embedding.parameters()},
                                     {'params': model.linear.parameters()},
                                     {'params': model.bias, 'weight_decay': 0}                                              
                                     ], 
                                      lr=args.lr, weight_decay=args.wdcy)                          

    best_loss = 1e10  
    test_auc = 0
    test_loss = 1e10  
    for epoch_i in range(args.epoch):
        train(model, optimizer, train_data_loader, criterion, device, norm_reg, args.reg_freq)
        auc,loss = test(model, train_data_loader, criterion, device)
        print("train_auc:",auc,"train_logloss:",loss)
        auc,loss = test(model, valid_data_loader, criterion, device)
        print("valid_auc:",auc,"valid_logloss:",loss)
        if loss<best_loss:
            best_loss = loss
            best_auc = auc
            test_auc,test_loss = test(model, test_data_loader, criterion, device) 
        else:
            break
    return(epoch_i, test_auc, test_loss)        



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train and test the field-wise learning model")
    parser.add_argument('--dataset-path', type=str, default=None, help="path to the dataset")
    parser.add_argument('--ebd-dim', type=float, default=8, help="embedding dimension")
    parser.add_argument('--log-ebd', action='store_true', default=False, help="whether to use log scale for embedding dimensions")    
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate for AdaGrad optimiser")
    parser.add_argument('--wdcy', type=float, default=1e-8, help="weight decay")  
    parser.add_argument('--include-linear', action='store_true', default=False, 
                        help="whether to include a linear term")  # if true, this is equal to add bias terms b_i^j-s to all models  
    parser.add_argument('--reg-lr', type=float, default=1e-10, help="learning rate for the regularisation terms")
    parser.add_argument('--reg-mean', action='store_true', default=False, help="whether to regularise the mean vector (column average of W)")  
    parser.add_argument('--reg-adagrad', action='store_true', default=False, help="whether to use AdaGrad for minimising the regularisation terms")
    parser.add_argument('--reg-sqrt', action='store_true', default=False, 
                        help="whether to apply square root on each regularisation term") # if True, the regularisation terms are directly related to the error bound
    parser.add_argument('--reg-freq', type=int, default=1000, 
                        help="considering the regularisation term every reg-freq iterations")                   
    parser.add_argument('--batch-size', type=int, default=2048, help="batch size")
    parser.add_argument('--epoch', type=int, default=100, help="max training epochs")
    parser.add_argument('--device', type=str, default="cuda:0", help="device to use")                   
    args = parser.parse_args()
    print(args)    
    
    test_results  = main(args)
    print("epoch: {0}, best_auc: {1}, best_loss: {2}".format(test_results[0],test_results[1],test_results[2]))
    
    
