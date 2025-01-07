import numpy as np
import os
import math
import nni
#import seaborn as sns
#import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def evaluate_location(y, top_k_pred, K=20):
    '''
    get hit ratio, mrr
    :param y: (batch,)
    :param top_k_pred: (batch, num_class)
    :param K
    :return:
    '''
    total_num = top_k_pred.shape[0]
    hit_ratio = np.zeros(K)
    mrr = []
    for i in range(total_num):  
        rank = np.where(top_k_pred[i] == y[i])[0] + 1
        mrr.append(rank)
        for j in range(1, K+1):   
            if y[i] in set(top_k_pred[i, :j]):
                hit_ratio[j-1] = hit_ratio[j-1] + 1
    hit_ratio = hit_ratio/total_num
    # print('mrr:',mrr)
    mrr = (1/np.array(mrr)).mean()
    return hit_ratio, mrr


def get_total_prob_c(loader, model, gts, gss, time_info, week_info, feature_category, feature_lat, feature_lng, feature_lat_ori, feature_lng_ori, save_filename=None, params_path=None, distance=None):
    '''
    calculates the loss, mae and mape for the entire data loader
    :param loader:
    :param save:
    :return:
    '''
    all_topic = []
    all_label = []
    for input in loader:
        topic, gamma_c = model.get_gammac(input, gss, feature_category, feature_lat, feature_lng, feature_lat_ori, feature_lng_ori, gts, time_info, week_info, distance=distance)  # (batch_size,), (batch_size,)
        all_topic.append(topic.detach().cpu().numpy())
        all_label.append(gamma_c.detach().cpu().numpy())
    all_topic = np.concatenate(all_topic)
    all_label = np.concatenate(all_label)
    all_label_index = np.argmax(all_label, axis=1)
    print('all_label:', all_label[:10])
    print('all_label_index:', all_label_index[:10])
    if save_filename is not None:
        filename = os.path.join(params_path, save_filename + '_gammac.npz')
        np.savez(filename, all_topic=all_topic, all_label=all_label, all_label_index=all_label_index)
    return all_topic, all_label, all_label_index


def density_visualization(density, ground_truth, batch_cnt):
    '''
    plot the probability density function
    :param density:
    :param ground_truth:
    :return:
    '''
    n_samples = 1000
    hours = 48
    x = np.linspace(0, hours, n_samples)
    t = ground_truth
    cnt = 0
    length = len(density)
    '''
    for i in range(length):
        y = density[i]
        plt.plot(x, y, "r-", label="STDGN")
        plt.legend()
        plt.xlabel(r"$\tau$", fontdict={'family': 'Times New Roman', 'size':16})
        plt.ylabel(r"p($\tau$)", fontdict={'family': 'Times New Roman', 'size':16})
        plt.yticks(fontproperties = 'Times New Roman', size = 14)
        plt.xticks(fontproperties = 'Times New Roman', size = 14)
        plt.grid()
        # plt.title("the probability density function in JKT dataset", fontdict={'family': 'Times New Roman', 'size':16})
        true_value = round(t[i], 2)
        plt.axvline(x=true_value, ls=":", c="black")
        plt.text(x=true_value + 1, y=1/2*np.max(y), s=r"$\tau_{n+1}$=" + str(true_value), size=16, alpha=0.8)
        plt.legend(prop={'family' : 'Times New Roman', 'size' : 16})
        plt.show()
        pic_name = str(batch_cnt) + '_' + str(cnt)
        cnt += 1
        plt.savefig(f'./data/density/jkt_{pic_name}.png')
        plt.savefig(f'./data/density/jkt_{pic_name}.eps',format='eps', dpi=10000)
        plt.close()
    '''


def softmax(x):
    '''
    self-define softmax operation
    :param x:
    :return:
    '''
    # print("before: ", x)
    x -= np.max(x, axis=1, keepdims=True)  # for stationary computation
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)  # formula
    # print("after: ", x)
    return x


def rad(d):
    '''
    rad the latitude and longitude
    :param d: latitude or longitude
    :return rad:
    '''
    return d * math.pi / 180.0


def getDistance(lat1, lng1, lat2, lng2):
    '''
    get the distance between two location using their latitude and longitude
    :param lat1:
    :param lng1:
    :param lat2:
    :param lng2:
    :return s:
    '''
    EARTH_REDIUS = 6378.137
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a/2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b/2), 2)))
    s = s * EARTH_REDIUS
    return s


def get_s_baselines_total_loss_s_for_MobilityLLM_RNN(loader, model, save_filename=None, params_path=None):
    all_loss_s = []
    all_ground_truth_location = []
    all_predicted_topK = []

    for input in loader:
        s_loss_score, top_k_pred = model(input)
        all_loss_s.append(s_loss_score.detach().cpu().numpy())
        all_ground_truth_location.append(input.Y_location.cpu().numpy())
        all_predicted_topK.append(top_k_pred.cpu().numpy())

    all_loss_s = np.array(all_loss_s)
    all_loss_s = np.mean(all_loss_s)

    all_ground_truth_location = np.concatenate(all_ground_truth_location)
    all_predicted_topK = np.concatenate(all_predicted_topK)

    hit_ratio, mrr = evaluate_location(all_ground_truth_location, all_predicted_topK)

    if save_filename is not None:
        filename = os.path.join(params_path, save_filename + '_results.npz')
        np.savez(filename, all_ground_truth_location=all_ground_truth_location, all_predicted_topK=all_predicted_topK)

    return all_loss_s, hit_ratio, mrr


# for downstream validation bencnmark
def get_s_baselines_total_loss_s_for_MobilityLLM_DEMO_DOWN(loader, model, downstream='POI', save_filename=None, params_path=None):
    all_loss_s = []
    all_ground_truth_users = []
    all_predicted_topK = []
    for input in loader:
        s_loss_score, top_k_pred = model(input, mode='downstream', downstream=downstream)
        all_loss_s.append(s_loss_score.detach().cpu().numpy())
        if downstream == 'POI':
            all_ground_truth_users.append(input.Y_location.cpu().numpy())
            # all_ground_truth_users.append(torch.index_select(torch.tensor(input.Y_location), dim=0, index=indice).cpu().numpy())
        elif downstream == 'TUL':
            all_ground_truth_users.append(input.X_users.cpu().numpy())
        else:
            raise ValueError('downstream is not in [POI, TUL]')

        all_predicted_topK.append(top_k_pred.cpu().numpy())

    all_loss_s = np.array(all_loss_s)
    all_loss_s = np.mean(all_loss_s)

    all_ground_truth_users = np.concatenate(all_ground_truth_users)
    all_predicted_topK = np.concatenate(all_predicted_topK)

    hit_ratio, mrr = evaluate_location(all_ground_truth_users, all_predicted_topK)

    if save_filename is not None:
        filename = os.path.join(params_path, save_filename + '_results.npz')
        np.savez(filename, all_ground_truth_users=all_ground_truth_users, all_predicted_topK=all_predicted_topK)

    return all_loss_s, hit_ratio, mrr


# for downstream validation bencnmark
def get_s_baselines_total_loss_s_for_MobilityLLM_DOWN(loader, model, downstream='POI', save_filename=None, params_path=None):
    all_loss_s = []
    all_ground_truth_users = []
    all_predicted_topK = []
    for input in loader:
        if input.X_all_loc.shape[1] >= 700:
            continue
        s_loss_score, top_k_pred, _ = model(input, mode='downstream', downstream=downstream)
        all_loss_s.append(s_loss_score.detach().cpu().numpy())
        if downstream == 'POI':
            all_ground_truth_users.append(input.Y_location.cpu().numpy())
        elif downstream == 'TUL':
            all_ground_truth_users.append(input.X_users.cpu().numpy())
        else:
            raise ValueError('downstream is not in [POI, TUL]')

        all_predicted_topK.append(top_k_pred.cpu().numpy())

    all_loss_s = np.array(all_loss_s)
    all_loss_s = np.mean(all_loss_s)

    all_ground_truth_users = np.concatenate(all_ground_truth_users)
    all_predicted_topK = np.concatenate(all_predicted_topK)

    hit_ratio, mrr = evaluate_location(all_ground_truth_users, all_predicted_topK)

    if save_filename is not None:
        filename = os.path.join(params_path, save_filename + '_results.npz')
        np.savez(filename, all_ground_truth_users=all_ground_truth_users, all_predicted_topK=all_predicted_topK)

    return all_loss_s, hit_ratio, mrr


def get_t_for_IFLTPP(loader, model, save_filename=None, params_path=None, experiment_base_dir=None, use_nni=False):
    '''
    calculates the loss, mae and mape for the entire data loader
    :param loader:
    :param save:
    :return:
    '''
    ground_truth_Y_tau = []
    predicted_Y_tau = []
    all_tnll_t = []
    all_nll = []

    for input in loader:
        tnll, mean, _ = model(input, cont_conf=None, mode='downstream', downstream='TPP')  # (batch_size,), (batch_size,)
        ground_truth_Y_tau.append(model.truth_Y_tau.detach().cpu().numpy())
        predicted_Y_tau.append(mean.detach().cpu().numpy())
        all_tnll_t.append(tnll.detach().cpu().numpy())

    all_tnll_t = np.array(all_tnll_t)

    ground_truth_Y_tau = np.concatenate(ground_truth_Y_tau).flatten()
    predicted_Y_tau = np.concatenate(predicted_Y_tau).flatten()

    mae = np.mean(abs(ground_truth_Y_tau - predicted_Y_tau))
    rmse = np.sqrt(((ground_truth_Y_tau - predicted_Y_tau) ** 2).mean())
    mape = np.mean(abs(ground_truth_Y_tau - predicted_Y_tau) / np.maximum(np.mean(ground_truth_Y_tau), ground_truth_Y_tau))
    tnll_t = np.mean(all_tnll_t)

    if (save_filename is not None) and (not use_nni):
        filename = os.path.join(params_path, save_filename+'_results.npz')
        np.savez(filename, ground_truth_Y_tau=ground_truth_Y_tau, predicted_Y_tau=predicted_Y_tau)

    return mae, mape, rmse, tnll_t


def get_semantic_information(cnt2category, data_root):
    import pickle
    vecpath = data_root + "glove.twitter.27B.50d.pkl"
    pkl_data = open(vecpath, "rb")
    word_vec = pickle.load(pkl_data)
    for word in word_vec.keys():
        word_vec[word] = word_vec[word]
    pkl_data.close()

    word_id = 0
    dataset_word_vec = []
    dataset_word_index = {} 
    categories = cnt2category.values()
    for category in categories:
        words = category.split(" ")
        # print(words)
        for word in words:
            word = word.lower()
            if (word in word_vec) and (word not in dataset_word_index): 
                dataset_word_index[word] = word_id
                word_id += 1
                dataset_word_vec.append(word_vec[word])
    print("word_index: ", dataset_word_index)
    return dataset_word_vec, dataset_word_index, word_id


