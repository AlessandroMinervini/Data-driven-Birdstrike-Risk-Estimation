import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import matplotlib
matplotlib.verbose = False
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy import signal, stats
from sklearn.metrics import mean_squared_error
from torch.optim.lr_scheduler import StepLR


# Manage files on drive
from google.colab import drive
drive.mount('/content/drive')


# Embedding configurations
config = {
    'species_embed': 256,
    'meteo_embed': 32,
    'temperature_embed': 32,
    'd_wind_embed': 16,
    'i_wind_embed': 16,
    'd_runway_embed': 256,
    'ambiente_embed': 32,
    'hidden_size': 512,
    'window': 15
    'offset': 7
}

# Variables.
batch_size = 128
epochs = 1000
lr = 0.00001
split_size = 75
out_lstm = 7680 #15360 #7680 15 days
lambda_l2 = 0.001
offset = config['offset']

step_decay = 200
lr_decay = 0.75

path = '/content/drive/My Drive/Tesi/dataset/AIRPORT-NAME/'

# Load features.
train_s = np.load(path + 'train_s.npy')
test_s = np.load(path + 'test_s.npy')
train_m = np.load(path + 'train_m.npy')
test_m = np.load(path + 'test_m.npy')
train_t = np.load(path + 'train_t.npy')
test_t = np.load(path + 'test_t.npy')
train_i_w = np.load(path + 'train_i_w.npy')
test_i_w = np.load(path + 'test_i_w.npy')
train_d_w = np.load(path + 'train_d_w.npy')
test_d_w = np.load(path + 'test_d_w.npy')
train_d_r = np.load(path + 'train_d_r.npy')
test_d_r = np.load(path + 'test_d_r.npy')
train_am = np.load(path + 'train_am.npy')
test_am = np.load(path + 'test_am.npy')
train_sb = np.load(path + 'train_sb.npy')
test_sb = np.load(path + 'test_sb.npy')
# Load BS back 1 year.
train_BS_1y = np.load(path + 'train_BS_back1year.npy')
test_BS_1y = np.load(path + 'test_BS_back1year.npy')
# Load BS.
train_BS = np.load(path + 'train_BS.npy')
test_BS = np.load(path + 'test_BS.npy')
# Load ground truth.
train_GT = np.load(path + 'train_GT.npy')
test_GT = np.load(path + 'test_GT.npy')


norm = MinMaxScaler(feature_range=(0, 1)).fit(train_s)
train_s = norm.transform(train_s)

norm = MinMaxScaler(feature_range=(0, 1)).fit(train_m)
train_m = norm.transform(train_m)

norm = MinMaxScaler(feature_range=(0, 1)).fit(train_t)
train_t = norm.transform(train_t)

norm = MinMaxScaler(feature_range=(0, 1)).fit(train_i_w)
train_i_w = norm.transform(train_i_w)

norm = MinMaxScaler(feature_range=(0, 1)).fit(train_d_w)
train_d_w = norm.transform(train_d_w)

norm = MinMaxScaler(feature_range=(0, 1)).fit(train_d_r)
train_d_r = norm.transform(train_d_r)

norm = MinMaxScaler(feature_range=(0, 1)).fit(train_am)
train_am = norm.transform(train_am)

norm = MinMaxScaler(feature_range=(0, 1)).fit(test_s)
test_s = norm.transform(test_s)

norm = MinMaxScaler(feature_range=(0, 1)).fit(test_m)
test_m = norm.transform(test_m)

norm = MinMaxScaler(feature_range=(0, 1)).fit(test_t)
test_t = norm.transform(test_t)

norm = MinMaxScaler(feature_range=(0, 1)).fit(test_i_w)
test_i_w = norm.transform(test_i_w)

norm = MinMaxScaler(feature_range=(0, 1)).fit(test_d_w)
test_d_w = norm.transform(test_d_w)

norm = MinMaxScaler(feature_range=(0, 1)).fit(test_d_r)
test_d_r = norm.transform(test_d_r)

norm = MinMaxScaler(feature_range=(0, 1)).fit(test_am)
test_am = norm.transform(test_am)


train_s = torch.tensor(train_s).float()
train_m = torch.tensor(train_m).float()
train_t = torch.tensor(train_t).float()
train_i_w = torch.tensor(train_i_w).float()
train_d_w = torch.tensor(train_d_w).float()
train_d_r = torch.tensor(train_d_r).float()
train_am = torch.tensor(train_am).float()
train_sb = torch.tensor(train_sb.reshape(-1,1)).float()
train_GT = torch.tensor(train_GT).float()
train_BS = torch.tensor(train_BS.reshape(-1,1)).float()
train_BS_1y = torch.tensor(train_BS_1y.reshape(-1,1)).float()

test_s = torch.tensor(test_s).float()
test_m = torch.tensor(test_m).float()
test_t = torch.tensor(test_t).float()
test_i_w = torch.tensor(test_i_w).float()
test_d_w = torch.tensor(test_d_w).float()
test_d_r = torch.tensor(test_d_r).float()
test_am = torch.tensor(test_am).float()
test_sb = torch.tensor(test_sb.reshape(-1,1)).float()
test_GT = torch.tensor(test_GT).float()
test_BS = torch.tensor(test_BS.reshape(-1,1)).float()
test_BS_1y = torch.tensor(test_BS_1y.reshape(-1,1)).float()


class TemporalTensorDataset(data.Dataset):
    def __init__(self, s, m, t, i_w, d_w, d_r, am, sb, bs, bs_1y, gt, window=1, sampling=1):
        self.specie = s
        self.meteo = m
        self.temperature = t
        self.i_wind = i_w
        self.d_wind = d_w
        self.d_runway = d_r
        self.am = am
        self.sb = sb
        self.GT = gt
        self.BS = bs
        self.BS_1y = bs_1y
        self.window = window
        self.sampling = sampling

    def __getitem__(self, index):
        y = self.GT[index // self.sampling: index // self.sampling + self.window + 1] # prendo il giorno k+1
        return (self.specie[index//self.sampling: index//self.sampling + self.window],
                self.meteo[index // self.sampling: index // self.sampling + self.window],
                self.temperature[index // self.sampling: index // self.sampling + self.window],
                self.i_wind[index // self.sampling: index // self.sampling + self.window],
                self.d_wind[index // self.sampling: index // self.sampling + self.window],
                self.d_runway[index // self.sampling: index // self.sampling + self.window],
                self.am[index // self.sampling: index // self.sampling + self.window],
                self.sb[index // self.sampling: index // self.sampling + self.window],
                self.BS[index // self.sampling: index // self.sampling + self.window],
                self.BS_1y[index // self.sampling: index // self.sampling + self.window],
                y[-1])

    def __len__(self):
        return (self.specie.size(0)-self.window+1)//self.sampling


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        # Get configuration parameters.
        self.hidden_size = config['hidden_size']
        self.species_embed = config['species_embed']
        self.meteo_embed = config['meteo_embed']
        self.temperature_embed = config['temperature_embed']
        self.d_wind_embed = config['d_wind_embed']
        self.i_wind_embed = config['i_wind_embed']
        self.d_runway_embed = config['d_runway_embed']
        self.window = config['window']
        self.ambiente_embed = config['ambiente_embed']
        self.suolo_b_embed = 1
        self.bs_size = 1
        self.bs_1y_size = 1

        self.embed_size = self.species_embed + self.meteo_embed + self.d_wind_embed + self.i_wind_embed +\
                          self.d_runway_embed + self.temperature_embed + self.ambiente_embed +\
                          self.suolo_b_embed + self.bs_size + self.bs_1y_size
        # Embedding layers.
        self.species_embedding = nn.Linear(train_s.shape[1], self.species_embed)
        self.meteos_embedding = nn.Linear(train_m.shape[1], self.meteo_embed)
        self.temperature_embedding = nn.Linear(train_t.shape[1], self.temperature_embed)
        self.d_wind_embedding = nn.Linear(train_d_w.shape[1], self.d_wind_embed)
        self.i_wind_embedding = nn.Linear(train_i_w.shape[1], self.i_wind_embed)
        self.d_runway_embedding = nn.Linear(train_d_r.shape[1], self.d_runway_embed)
        self.ambiente_embedding = nn.Linear(train_am.shape[1], self.ambiente_embed)
        # LSTM.
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True, num_layers=2, dropout=0.3)
        # Linear.
        self.output = nn.Linear(out_lstm, 1)

    def forward(self, species_f, meteo_f, temperature_f, i_wind_f, d_wind_f, d_runway_f, ambiente_f, sb_f, bs_f, bs_1y_f):
        batch_size_ = batch_size//2
        # Embedding data.
        species_embed = self.species_embedding(species_f)
        meteo_embed = self.meteos_embedding(meteo_f)
        temperature_embed = self.temperature_embedding(temperature_f)
        i_wind_embed = self.i_wind_embedding(i_wind_f)
        d_wind_embed = self.d_wind_embedding(d_wind_f)
        d_runway_embed = self.d_runway_embedding(d_runway_f)
        ambiente_embed = self.ambiente_embedding(ambiente_f)
        # Concat data.
        embedding = torch.cat((species_embed, meteo_embed, temperature_embed,
                               i_wind_embed, d_wind_embed, d_runway_embed, ambiente_embed, sb_f, bs_f, bs_1y_f), -1)
        # Reshaping data for LSTM layer.
        embedding = embedding.view(batch_size_, self.window, -1)
        # LSTM + linear layer
        output, _ = self.lstm(F.relu(embedding))
        output = F.relu(output)
        output = output.reshape(batch_size_, -1)
        output = F.sigmoid(self.output(output))
        return output
    
def MRL(pred_1, pred_2, labels):
  loss_fn = nn.MarginRankingLoss(margin=0)
  loss = loss_fn(pred_1, pred_2, labels)
  return loss

def L1loss(output, y): 
    loss_fn = nn.L1Loss()
    loss = loss_fn(output, y)
    return loss

def data_plot(data, gt, title=''):
    X = []
    for i in range(len(data)):
        X.append(i)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(X, gt[offset:len(data)+offset], label= 'GT')
    #ax.plot(X, gt[0:len(data)], label= 'GT')
    ax.plot(X, data, label='y')
    ax.set(xlabel='Days', ylabel='Risk-values', title='New Risk-index vs. Ground Truth')
    ax.grid()
    ax.legend()
    plt.show()

def prep_data(s, m, t, i_w, d_w, d_r, am, sb, bs, bs_1y, y):
    bsize = batch_size//2

    s_x1 = s[bsize:]
    s_x2 = s[0:bsize]
    m_x1 = m[bsize:]
    m_x2 = m[0:bsize]
    t_x1 = t[bsize:]
    t_x2 = t[0:bsize]
    i_w_x1 = i_w[bsize:]
    i_w_x2 = i_w[0:bsize]
    d_w_x1 = d_w[bsize:]
    d_w_x2 = d_w[0:bsize]
    d_r_x1 = d_r[bsize:]
    d_r_x2 = d_r[0:bsize]
    am_x1 = am[bsize:]
    am_x2 = am[0:bsize]
    sb_x1 = sb[bsize:]
    sb_x2 = sb[0:bsize]
    bs_x1 = bs[bsize:]
    bs_x2 = bs[0:bsize]
    bs_1y_x1 = bs_1y[bsize:]
    bs_1y_x2 = bs_1y[0:bsize]
    y1 = y[bsize:]
    y2 = y[0:bsize]
    return s_x1, s_x2, m_x1, m_x2, t_x1, t_x2, i_w_x1, i_w_x2, d_w_x1, d_w_x2, d_r_x1, d_r_x2, am_x1, am_x2, sb_x1, sb_x2, bs_x1, bs_x2, bs_1y_x1, bs_1y_x2, y1, y2

def compute_labels(Y1, Y2):
  labels = torch.zeros(Y1.shape[0], Y1.shape[1])
  labels[Y1 > Y2] = 1
  labels[Y1 <= Y2] = -1
  return labels

def customLoss(marginL, L1, L2):
  sigma = 0.2
  sigma_2 = 1
  marginL = marginL.cuda()
  L1 = L1.cuda()
  L2 = L2.cuda()
  loss = sigma_2*marginL + (sigma)*(L1 + L2)
  return loss

# Model.
model = Model(config)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_l2)
scheduler = StepLR(optimizer, step_size=step_decay, gamma=lr_decay)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

trainset = TemporalTensorDataset(train_s, train_m, train_t, train_i_w, train_d_w, train_d_r, train_am, train_sb, train_BS, train_BS_1y, train_GT, config['window'], 1)
testset = TemporalTensorDataset(test_s, test_m, test_t, test_i_w, test_d_w, test_d_r, test_am, test_sb, test_BS, test_BS_1y, test_GT, config['window'], 1)

# Scores.
train_loss = []
test_loss = []
train_s_corr = []
test_s_corr = []
test_mse = []
print(model)


# Training.
dl_train = data.DataLoader(trainset, batch_size, shuffle=True, drop_last=True)
for epoch in range(epochs):
    scheduler.step()
    l_rate = get_lr(optimizer)
    print('Learning rate:', l_rate)
    train_vals = []
    print(f'Training epoch {epoch}...')
    model.train()
    total_loss = 0
    for (s, m, t, i_w, d_w, d_r, am, sb, bs, bs_1y, y) in dl_train:
        s_x1, s_x2, m_x1, m_x2, t_x1, t_x2, i_w_x1, i_w_x2, d_w_x1, d_w_x2, d_r_x1, d_r_x2, am_x1, am_x2, sb_x1, sb_x2, bs_x1, bs_x2, bs_1y_x1, bs_1y_x2, y1, y2 = prep_data(s, m, t, i_w, d_w, d_r, am, sb, bs, bs_1y, y)
        s_x1 = s_x1.cuda()
        s_x2 = s_x2.cuda()
        m_x1 = m_x1.cuda()
        m_x2 = m_x2.cuda()
        t_x1 = t_x1.cuda()
        t_x2 = t_x2.cuda()
        i_w_x1 = i_w_x1.cuda()
        i_w_x2 = i_w_x2.cuda()
        d_w_x1 = d_w_x1.cuda()
        d_w_x2 = d_w_x2.cuda()
        d_r_x1 = d_r_x1.cuda()
        d_r_x2 = d_r_x2.cuda()
        am_x1 = am_x1.cuda()
        am_x2 = am_x2.cuda()
        sb_x1 = sb_x1.cuda()
        sb_x2 = sb_x2.cuda()
        bs_x1 = bs_x1.cuda()
        bs_x2 = bs_x2.cuda()
        bs_1y_x1 = bs_1y_x1.cuda()
        bs_1y_x2 = bs_1y_x2.cuda()
        y1 = y1.cuda()
        y2 = y2.cuda()
        labels = compute_labels(y1, y2)
        model.zero_grad()
        output_1 = model.forward(s_x1, m_x1, t_x1, i_w_x1, d_w_x1, d_r_x1, am_x1, sb_x1, bs_x1, bs_1y_x1)
        output_2 = model.forward(s_x2, m_x2, t_x2, i_w_x2, d_w_x2, d_r_x2, am_x2, sb_x2, bs_x2, bs_1y_x2)
        MRL_loss = MRL(output_1.cpu(), output_2.cpu(), labels)
        L1_loss_1 = L1loss(output_1, y1)
        L1_loss_2 = L1loss(output_2, y2)
        loss = customLoss(MRL_loss, L1_loss_1, L1_loss_2)
        loss = loss.cuda()
        loss.backward()
        total_loss += loss.cpu().data.numpy()
        optimizer.step()
    print(f' Training loss: {total_loss / len(dl_train)}')
    train_loss.append(total_loss / len(dl_train))

# Testing.
    ys = []
    model.eval()
    total_loss = 0.0
    predictions = []
    dl_test = data.DataLoader(testset, batch_size//2, shuffle=False, drop_last=True)
    for (s, m, t, i_w, d_w, d_r, am, sb, bs, bs_1y, y) in dl_test:
        for i in y:
            ys.append(i)
        s = s.cuda()
        m = m.cuda()
        t = t.cuda()
        i_w = i_w.cuda()
        d_w = d_w.cuda()
        d_r = d_r.cuda()
        am = am.cuda()
        sb = sb.cuda()
        bs = bs.cuda()
        bs_1y = bs_1y.cuda()
        y = y.cuda()
        output = model.forward(s, m, t, i_w, d_w, d_r, am, sb, bs, bs_1y)
        for pred in output:
            predictions.append(pred.cpu().detach().numpy())
    #data_plot(ys, test_GT, title='Test Predictions')


# Display testing fitting.
    loader_ = data.DataLoader(trainset, batch_size//2, shuffle=False, drop_last=True)
    train_vals = []
    for (s, m, t, i_w, d_w, d_r, am, sb, bs, bs_1y, y) in loader_:
        s = s.cuda()
        m = m.cuda()
        t = t.cuda()
        i_w = i_w.cuda()
        d_w = d_w.cuda()
        d_r = d_r.cuda()
        am = am.cuda()
        sb = sb.cuda()
        bs = bs.cuda()
        bs_1y = bs_1y.cuda()
        y = y.cuda()
        output = model.forward(s, m, t, i_w, d_w, d_r, am, sb, bs, bs_1y)
        for pred in output:
            train_vals.append(pred.cpu().detach().numpy())

    # Print Spearman correlation.
    test_s_spr = stats.spearmanr(test_GT[offset:len(predictions)+offset], predictions)
    test_s_corr.append(test_s_spr[0])
    #print('---------------------------------------------------')
    print('SPEARMAN correlation: ', test_s_spr[0])


# Print Loss values
X = []
for i in range(epochs):
  X.append(i)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(X, train_loss)
ax.set(xlabel='Epochs', ylabel='Loss', title='Train Loss')
ax.grid()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(X, test_s_corr)
ax.set(xlabel='Epochs', ylabel='Spearman Correlation', title='Spearman Correlation')
ax.grid()


norm = MinMaxScaler(feature_range=(0, 1)).fit(predictions)
rescale_predictions = norm.transform(predictions)

data_plot(rescale_predictions, test_GT, title='Test Predictions')
plt.show()
