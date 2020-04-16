
import numpy as np
from sklearn import linear_model
import torch
from model import FairRepMulti
import torch.optim as optim
import torch.nn as nn
from IPython import embed
from helpers import normalize, cal_emd_resamp
from dumb_containers import evaluate_performance_sim


def main():
    with open('data/adult.data.processed','r') as f:
        data = np.array([[float(x) for x in y.split()] for y in f.readlines()])

    P_col = 7
    P = data[:, P_col]
    y = data[:, -1]
    X = data[:, :-1]

    X = normalize(X, 20)

    print('number of unique class in the protected attribute is {0}'.format(len(set(P))))

    model = FairRepMulti(len(X[0]), 10, len(set(P)))
    model.encoder = nn.Sequential(nn.Linear(len(X[0]),10),
                                  nn.ReLU(),
                                  nn.Linear(10,10))

    model.decoder = nn.Sequential(nn.Linear(10,10),
                                  nn.ReLU(),
                                  nn.Linear(10,13))

    lr = 0.01
    optim_encoder = optim.Adam(model.encoder.parameters(), lr=lr)
    optim_decoder = optim.Adam(model.decoder.parameters(), lr=lr)
    optim_critic = []
    for i, t in enumerate(model.critic):
        optim_critic.append(optim.Adam(model.critic[i].parameters(), lr=lr))

    num_epoch = 200
    batch_size = 1000

    X_groups = []
    P_uni = sorted(list(set(P)))
    for i, p in enumerate(P_uni):
        X_groups.append(X[P==p])

    X_groups_lens = list(map(len, X_groups))
    min_len_required = 5*batch_size
    for i, x in enumerate(X_groups_lens):
        if x < min_len_required:
            X_groups[i] = X_groups[i][
                np.random.choice(len(X_groups[i]), min_len_required)
            ]

    print('length of each group:')
    print(list(map(len, X_groups)))


    X_groups_lens = list(map(len, X_groups))
    X_size = sum(X_groups_lens)
    num_iter = int(X_size / batch_size) * num_epoch

    use_cuda = True
    if use_cuda:
        model.cuda()

    cur_batch_stop = np.zeros(len(P_uni)).astype(int)
    alpha = 1000

    print_interval = 200
    print('number of total iterations: ' + str(num_iter))
    wdists_catch = np.zeros(len(P_uni))

    for i_iter in range(num_iter):
        optim_decoder.zero_grad()
        optim_encoder.zero_grad()
        for op in optim_critic:
            op.zero_grad()

        i = int(i_iter/10) % len(P_uni)
        x_g = X_groups[i]

        right_stop = min(len(x_g), cur_batch_stop[i] + batch_size)

        x_batch = x_g[cur_batch_stop[i]: right_stop]
        cur_batch_stop[i] = right_stop % len(x_g)

        x_rest_idx = np.random.choice(
            np.arange(len(X))[P != P_uni[i]],
            len(x_batch))
        x_rest = X[x_rest_idx]

        x_batch = torch.tensor(x_batch).float()
        x_rest = torch.tensor(x_rest).float()
        if use_cuda:
            x_batch = x_batch.cuda()
            x_rest = x_rest.cuda()

        for _ in range(10):
            optim_critic[i].zero_grad()
            wdist_neg = -model.wdist(x_batch, x_rest, i)
            wdist_neg.backward(retain_graph=True)
            optim_critic[i].step()

            for pa in model.critic[i].parameters():
                pa.data.clamp_(-0.01, 0.01)

        mse, wdists = model.forward(x_batch, x_rest, i)
        wdists_catch[i] = wdists
        loss = mse + 1000 * wdists
        loss.backward(retain_graph=True)
        optim_encoder.step()
        optim_decoder.step()

        if i_iter % print_interval == print_interval-1:
            print('[{0}/{1}] mse: {2} wdists: [{3}]'.format(
                i_iter, num_iter, mse.item(),
                ' '.join([str(w.item()) for w in wdists_catch])
            ))

    X_torch = torch.tensor(X).float()
    if use_cuda:
        X_torch = X_torch.cuda()
    U = model.encoder(X_torch)
    U = U.data.cpu().numpy()
    del X_torch

    print("let's see origin one-vs-all emds.")
    for p in P_uni:
        x_p = X[P==p]
        x_rest = X[P!=p]
        print(cal_emd_resamp(x_p, x_rest, 100, 10))

    print("let's see afterward one-vs-all emds.")
    for p in P_uni:
        x_p = U[P==p]
        x_rest = U[P!=p]
        print(cal_emd_resamp(x_p, x_rest, 100, 10))

    print("let's see now the classification performance and statistical pairty.")
    print("all is performed on the whole training set.")

    lin_cls_ori = linear_model.LogisticRegression(C=0.1)
    lin_cls_adv = linear_model.LogisticRegression(C=0.1)

    train_test_split = int(0.7 * len(X))
    X_train = X[:train_test_split]
    U_train = U[:train_test_split]
    y_train = y[:train_test_split]
    P_train = P[:train_test_split]

    X_test = X[train_test_split+1:]
    U_test = U[train_test_split+1:]
    y_test = y[train_test_split+1:]
    P_test = P[train_test_split+1:]

    lin_cls_ori.fit(X_train, y_train)
    lin_cls_adv.fit(U_train, y_train)

    y_pred_ori = lin_cls_ori.predict_proba(X_test)[:,1]
    y_pred_adv = lin_cls_adv.predict_proba(U_test)[:,1]

    print("original performance (ks, recall, precision, f1):")
    print(evaluate_performance_sim(y_test, y_pred_ori))
    print("fair rep performance (ks, recall, precision, f1):")
    print(evaluate_performance_sim(y_test, y_pred_adv))

    print("P's: " + str(P_uni))
    avg_score_ori = []
    avg_score_adv = []
    for p in P_uni:
        avg_score_ori.append(1.0*y_pred_ori[P_test==p].sum()/(P_test==p).sum())
        avg_score_adv.append(1.0*y_pred_adv[P_test==p].sum()/(P_test==p).sum())
    print("original avg scores:")
    print(avg_score_ori)
    print("fair rep avg scores:")
    print(avg_score_adv)

    print("original parity: " + str(max(avg_score_ori)/min(avg_score_ori)))
    print("fair rep parity: " + str(max(avg_score_adv)/min(avg_score_adv)))


if __name__ == '__main__':
    main()
    embed()
