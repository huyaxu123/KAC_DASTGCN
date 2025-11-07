import logging
import os
import gc
import argparse
import math
import random
import warnings
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils



from script import dataloader, utility, earlystopping, opt
from model import models
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#import nni

def set_env(seed):
    # Set available CUDA devices

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_parameters():
    parser = argparse.ArgumentParser(description='KCA_DSTGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='Enable CUDA acceleration')
    parser.add_argument('--seed', type=int, default=35, help='Set random seed for reproducibility')
    parser.add_argument('--dataset', type=str, default='GTDATA', choices=['GTDATA'], help='Dataset for training and evaluation')
    parser.add_argument('--n_his', type=int, default=18, help='Number of historical time steps as input')
    parser.add_argument('--n_pred', type=int, default=1, help='Number of time intervals to predict')
    parser.add_argument('--Kt', type=int, default=2, help='Temporal convolution kernel size')
    parser.add_argument('--stblock_num', type=int, default=2, help='Number of spatio-temporal blocks')
    parser.add_argument('--act_func', type=str, default='gtu', choices=['glu', 'gtu'], help='Activation function type')
    parser.add_argument('--model', type=str, default='KCA_DSTGCN', choices=['lstm','cnn'], help='Type of graph convolution module')
    parser.add_argument('--enable_bias', type=bool, default=True, help='Enable bias term in layers')
    parser.add_argument('--droprate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.001, help='Weight decay (L2 regularization factor')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs ')
    parser.add_argument('--opt', type=str, default='adam', choices=['adamw', 'lion', 'tiger'], help='Optimizer type ')
    parser.add_argument('--step_size', type=int, default=20, help='Step size for learning rate decay')
    parser.add_argument('--gamma', type=float, default=0.95, help='Decay factor for learning rate scheduler ')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--numZ', type=int, default=3, help='Embedding dimension')
    parser.add_argument('--numN', type=int, default=9, help='Number of nodes in the graph')
    parser.add_argument('--gcn_bool', type=bool, default=True, help='Whether to include a graph convolution layer')
    parser.add_argument('--aptonly', type=bool, default=True, help='whether only adaptive adj')
    parser.add_argument('--addaptadj', type=bool, default=True, help='Whether to add a learnable adjacency matrix')
    parser.add_argument('--thr', type=float, default=0.75, help='Threshold value between 0 and 1')


    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda:0')
        torch.cuda.empty_cache() # Clean cache
    else:
        device = torch.device('cpu')
        gc.collect() # Clean cache
    
    blocks = [ [1], [64, 64],  [64, 64], [64, 128], [1]]   #网络层数
    stblock_num = args.stblock_num
    return args, device, blocks, stblock_num 

def data_preparate(args, device):
    graph_p = [
    [0, 1, 1, 0, 0, 1, 0, 0, 0],  # P1
    [1, 0, 1, 0, 0, 1, 0, 0, 0],  # T1
    [1, 1, 0, 0, 0, 1, 1, 0, 1],  # T2
    [0, 0, 0, 0, 1, 1, 0, 0, 0],  # T4
    [0, 0, 0, 1, 0, 1, 0, 0, 0],  # P4
    [1, 1, 1, 1, 1, 0, 1, 0, 0],  # WO
    [0, 0, 1, 0, 0, 1, 0, 1, 1],  # IGV
    [0, 0, 0, 0, 0, 0, 1, 0, 1],  # MF
    [0, 0, 1, 0, 0, 0, 1, 1, 0],  # LHV
]

    args.graph_p = torch.tensor(graph_p, dtype=torch.float32, device=device)

    n_vertex = args.numN
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, args.dataset)
    data_col = pd.read_csv(os.path.join(dataset_path, 'case2_GTDATA.csv'),encoding='gbk').shape[0]
    # recommended dataset split rate as train: val: test = 60: 20: 20, 70: 15: 15 or 80: 10: 10

    val_rate = 0.15
    test_rate = 0.15
    len_val = int(math.floor(data_col * val_rate))
    len_test = int(math.floor(data_col * test_rate))
    len_train = int(data_col - len_val - len_test)

    train, val, test = dataloader.load_data(args.dataset, len_train, len_val)
    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)
    x_train, y_train = dataloader.data_transform(train, args.n_his, args.n_pred, device)
    x_val, y_val = dataloader.data_transform(val, args.n_his, args.n_pred, device)
    x_test, y_test = dataloader.data_transform(test, args.n_his, args.n_pred, device)

    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    return n_vertex, zscore, train_iter, val_iter, test_iter

def prepare_model(args, blocks, n_vertex, stblock_num):
    loss = nn.MSELoss()
    es = earlystopping.EarlyStopping(delta=0.0, 
                                     patience=args.patience, 
                                     verbose=True, 
                                     path="KCA_DSTGCN_" + args.dataset + ".pt")

    if args.model == 'KCA_DSTGCN':
        model = models.KCA_DSTGCN(args, blocks, n_vertex, stblock_num).to(device)
    elif args.model == 'lstm':
        model = lstmmodels.LSTMMain(input_size=9, output_len=1,
                                  lstm_hidden=128, lstm_layers=2, batch_size=64, num_vertices=9).to(device)
    elif args.model == 'cnn':
        model = cnnmodels.TTConvBlock( Kt=3, n_vertex=9, last_block_channel=32, channels=32, act_func='relu',  droprate=0.5).to(device)       
    elif args.model == 'mlp':
        model = mlpmodels.MLPBlock(input_dim=108, hidden_dim=64, output_dim=9, act_func='relu').to(device)
    else:
        raise ValueError(f"Unsupported graph convolution type: {args.model}")

    if args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == 'lion':
        optimizer = opt.Lion(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == 'tiger':
        optimizer = opt.Tiger(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    else:
        raise ValueError(f'ERROR: The {args.opt} optimizer is undefined.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return loss, es, model, optimizer, scheduler

def train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter):
    train_time = []
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        t1 = time.time()
        for x, y in tqdm.tqdm(train_iter):
            optimizer.zero_grad()
            y_pred = model(x).view(len(x), -1)  # [batch_size, num_nodes]
            l = loss(y_pred, y)
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        t2 = time.time()
        train_time.append(t2 - t1)
        scheduler.step()
        val_loss = val(model, val_iter)
        
        
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))

        es(val_loss, model)
        if es.early_stop:
            print("Early stopping")
            break
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
@torch.no_grad()
def val(model, val_iter):
    model.eval()

    l_sum, n = 0.0, 0
    for x, y in val_iter:
        y_pred = model(x).view(len(x), -1)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)

@torch.no_grad() 
def atest(zscore, loss, model, test_iter, args):

    model.load_state_dict(torch.load("KCA_DSTGCN_GTDATA.pt"))
    model.eval()
    s1 = time.time()
    test_MSE = utility.evaluate_model(model, loss, test_iter)
    s2 = time.time()
    test_MAE, test_RMSE, test_WMAPE, all_y,  all_y_pred = utility.evaluate_metric(model, test_iter, zscore)

    print("Average Test Time: {:.4f} secs".format(s2-s1))
    print(f'Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')
    return all_y,  all_y_pred


def r2_score_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan  

def mape_np(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return np.nan
    nz = np.abs(y_true) > eps
    if not np.any(nz):
        return np.nan
    return np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100.0


if __name__ == "__main__":
    # Logging
    #logger = logging.getLogger('KCA_DSTGCN')
    #logging.basicConfig(filename='KCA_DSTGCN.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    args, device, blocks, stblock_num = get_parameters()
    n_vertex, zscore, train_iter, val_iter, test_iter = data_preparate(args, device)
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex, stblock_num)
    # train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter)
    all_y,  all_y_pred = atest(zscore, loss, model, test_iter, args)

    reshaped_all_y = []
    reshaped_all_y_pred = []
    # Reshape each batch to 2D
    for y_true in all_y:
        reshaped = np.reshape(y_true, (-1, 9)) 
        reshaped_all_y.append(reshaped)
    for y_true_pred in all_y_pred:
        reshaped1 = np.reshape(y_true_pred, (-1, 9))  
        reshaped_all_y_pred.append(reshaped1)

    all_batches_all_y = np.concatenate(reshaped_all_y, axis=0)
    all_batches_all_y_pred = np.concatenate(reshaped_all_y_pred, axis=0)
    df_y = pd.DataFrame(all_batches_all_y, columns=[f'True_Value_{i + 1}' for i in range(all_batches_all_y.shape[1])])
    df_y_pred = pd.DataFrame(all_batches_all_y_pred,
                             columns=[f'Predicted_Value_{i + 1}' for i in range(all_batches_all_y_pred.shape[1])])

    # Metrics for target columns
    target_cols = [2, 3, 4, 5]
    assert df_y.shape == df_y_pred.shape, "df_y and df_y_pred shape mismatch"

    metrics = []
    for col in target_cols:
        
        y_t = pd.to_numeric(df_y.iloc[:, col], errors="coerce").to_numpy()
        y_p = pd.to_numeric(df_y_pred.iloc[:, col], errors="coerce").to_numpy()

        r2 = r2_score_np(y_t, y_p)
        mape = mape_np(y_t, y_p)

        print(f"Col {col} -> R2 = {r2:.4f}, MAPE = {mape:.4f}%")
        metrics.append({"Column": col, "R2": r2, "MAPE(%)": mape})
    avg_r2 = np.nanmean([m["R2"] for m in metrics])
    avg_mape = np.nanmean([m["MAPE(%)"] for m in metrics])

    print(f"Average -> R2 = {avg_r2:.4f}, MAPE = {avg_mape:.4f}%")

    