import torch.utils.data as data_utils
import os
import itertools
import numpy as np
import torch


def tid_list_48(tm):
    if tm.weekday() in [0, 1, 2, 3, 4]:
        tid = int(tm.hour)
    else:
        tid = int(tm.hour) + 24
    return tid


def load_data_from_dataset(set_name, loader, device, user_cnt, venue_cnt, save_split, name, data_root):
    X_target_lengths = loader[f'{set_name}X_target_lengths']
    X_arrival_times = loader[f'{set_name}X_arrival_times']
    X_users = loader[f'{set_name}X_users']
    X_locations = loader[f'{set_name}X_locations']
    X_locations_category = loader[f'{set_name}X_locations_category']
    X_lats = loader[f'{set_name}X_lats']
    X_lons = loader[f'{set_name}X_lons']
    X_geohash = loader[f'{set_name}X_geohash']
    Y_location = loader[f'{set_name}Y_locations']
    Y_location_category = loader[f'{set_name}Y_locations_category']
    Y_lat = loader[f'{set_name}Y_lats']
    Y_lon = loader[f'{set_name}Y_lons']
    Y_geohash = loader[f'{set_name}Y_geohash']

    X_tau = pad_1d(loader[f'{set_name}X_delta_times'])
    Y_tau = pad_1d(loader[f'{set_name}Y_delta_times'])

    X_all_loc = []
    X_all_loc_category = []
    X_all_lat = []
    X_all_lon = []
    X_all_geohash = []
    X_all_tim = []
    X_lengths = []
    for i in range(len(X_arrival_times)):
        tim = X_arrival_times[i]
        loc = X_locations[i]
        loc_category = X_locations_category[i]
        lat = X_lats[i]
        lon = X_lons[i]
        geohash = X_geohash[i]

        len_ = len(tim)
        for j in range(len_):
            tim[j] = tid_list_48(tim[j])

        X_all_loc.append(loc)
        X_all_loc_category.append(loc_category)
        X_all_lat.append(lat)
        X_all_lon.append(lon)
        X_all_geohash.append(geohash)
        X_all_tim.append(tim)
        X_lengths.append(len_)

    print("X_all_loc: ", len(X_all_loc), X_all_loc[0])
    print("X_all_loc_category: ", len(X_all_loc), X_all_loc_category[0])
    print("X_all_lat: ", len(X_all_loc), X_all_lat[0])
    print("X_all_lon: ", len(X_all_loc), X_all_lon[0])
    print("X_all_geohash: ", len(X_all_loc), X_all_geohash[0])
    print("X_all_tim: ", len(X_all_tim), X_all_tim[0])
    print("X_target_lengths: ", len(X_target_lengths), X_target_lengths[0])
    print("X_lengths: ", len(X_lengths), X_lengths[0])
    print("X_users:", len(X_users), X_users)
    print("Y_location:", len(Y_location), Y_location[0])
    print("Y_location_category:", len(Y_location), Y_location_category[0])
    print("Y_lat:", len(Y_location), Y_lat[0])
    print("Y_lon:", len(Y_location), Y_lon[0])
    print("Y_geohash:", len(Y_location), Y_geohash[0])

    dataset = SessionBasedSequenceDataset(device, user_cnt, venue_cnt, X_users, X_all_loc, X_all_loc_category,
                                          X_all_lat, X_all_lon, X_all_geohash,
                                          X_all_tim, Y_location, Y_location_category, Y_lat, Y_lon, Y_geohash,
                                          X_target_lengths, X_lengths, None, X_tau, Y_tau)
    print(f'samples cnt of data_{set_name}:', dataset.real_length())

    return dataset


def load_dataset_for_MobilityLLM(name, data_root, save_split=False, device=None):
    '''
    1. load data and construct train/val/test dataset
    2. construct temporal graphs gts and spatial graphs gss
    3. construct SessionBasedSequenceDataset
    :param name: file name
    :param log_mode: whether log(X_taus), default True
    :return:
    '''
    if not name.endswith('.npz'):
        name += '.npz'

    if save_split:
        train_loader = dict(np.load(os.path.join(data_root + 'new_datasets', 'train_' + name), allow_pickle=True))
        val_loader = dict(np.load(os.path.join(data_root + 'new_datasets', 'val_' + name), allow_pickle=True))
        loader = dict(np.load(os.path.join(data_root + 'new_datasets', 'test_' + name), allow_pickle=True))
        category_vector = np.load(os.path.join(data_root + 'new_datasets', 'category_' + name))

    else:
        loader = dict(np.load(os.path.join(data_root, 'test_' + name), allow_pickle=True))
        train_loader = loader
        val_loader = loader
        category_vector = None

    user_cnt = loader['user_cnt']
    venue_cnt = loader['venue_cnt']
    print('user_cnt:', user_cnt)
    print('venue_cnt:', venue_cnt)

    feature_category = loader['feature_category']
    feature_lat = loader['feature_lat']  # index
    feature_lng = loader['feature_lng']  # index

    # put spatial point features into tensor
    feature_category = torch.LongTensor(feature_category)
    feature_lat = torch.LongTensor(feature_lat)
    feature_lng = torch.LongTensor(feature_lng)

    latN, lngN = loader['latN'], loader['lngN']
    category_cnt = loader['category_cnt']

    # ----- load train / val / test to get dataset -----
    data_train = load_data_from_dataset('train', train_loader, device, user_cnt, venue_cnt, save_split, name, data_root)
    data_val = load_data_from_dataset('val', val_loader, device, user_cnt, venue_cnt, save_split, name, data_root)
    data_test = load_data_from_dataset('test', loader, device, user_cnt, venue_cnt, save_split, name, data_root)

    return data_train, data_val, data_test, feature_category, feature_lat, feature_lng, latN, lngN, category_cnt, \
           category_vector['categories']


class SessionBasedSequenceDataset(data_utils.Dataset):
    """Dataset class containing variable length sequences.
    """

    def __init__(self, device, user_cnt, venue_cnt, X_users, X_all_loc, X_all_loc_category, X_all_lat, X_all_lon,
                 X_all_geohash,
                 X_all_tim, Y_location, Y_location_category, Y_lat, Y_lon, Y_geohash, target_lengths, X_lengths,
                 X_all_text,
                 X_tau, Y_tau):
        # torch.set_default_tensor_type(torch.cuda.FloatTensor)
        self.device = device
        self.user_cnt = user_cnt
        self.venue_cnt = venue_cnt
        self.X_users = X_users
        self.X_all_loc = X_all_loc
        self.X_all_loc_category = X_all_loc_category
        self.X_all_lat = X_all_lat
        self.X_all_lon = X_all_lon
        self.X_all_geohash = X_all_geohash
        self.X_all_tim = X_all_tim
        self.target_lengths = target_lengths
        self.X_lengths = X_lengths
        self.Y_location = Y_location
        self.Y_location_category = Y_location_category
        self.Y_lat = Y_lat
        self.Y_lon = Y_lon
        self.Y_geohash = Y_geohash
        self.X_all_text = X_all_text

        self.X_tau = [torch.Tensor(_) for _ in X_tau]
        # self.Y_tau = torch.Tensor(Y_tau.astype('float64')) / 60  # mins->hour
        self.Y_tau = torch.Tensor(Y_tau.astype('float64'))  # mins

        self.validate_data()

    @property
    def num_series(self):
        return len(self.Y_location)

    def real_length(self):
        res = 0
        n = len(self.Y_location)
        for i in range(n):
            res += len(self.Y_location[i])
        return res

    def validate_data(self):
        if len(self.X_all_loc) != len(self.Y_location) or len(self.X_all_tim) != len(self.Y_location):
            raise ValueError("Length of X_all_loc, X_all_tim, Y_location should match")

    def get_tau_log_mean_std_Y(self):
        """Get mean and std of Y_taus."""
        y = torch.flatten(self.Y_tau)
        logy = y[y != 0].log()
        return logy.mean(), logy.std()

    def get_mean_std_Y_tau(self):
        """Get mean and std of Y_tau."""
        y = torch.flatten(self.Y_tau)
        y = y[y != 0]
        return y.mean(), y.std()

    def normalize_Y_tau(self, mean=None, std=None):
        self.Y_tau = (self.Y_tau - mean)/std
        return self

    def __getitem__(self, key):
        '''
        the outputs are feed into collate()
        :param key:
        :return:
        '''
        return self.X_all_loc[key], self.X_all_tim[key], None, self.Y_location[key], self.target_lengths[key], \
               self.X_lengths[key], self.X_users[key], self.X_all_loc_category[key], self.X_all_lat[key], \
               self.X_all_lon[key], self.Y_location_category[key], self.Y_lat[key], self.Y_lon[key], \
               self.X_all_geohash[key], self.Y_geohash[key], self.X_tau[key], self.Y_tau[key], self.device

    def __len__(self):
        return self.num_series

    def __repr__(self):
        pass


def pad_session_data_one(data):
    fillvalue = 0
    # zip_longest
    data = list(zip(*itertools.zip_longest(*data, fillvalue=fillvalue)))
    res = []
    res.extend([list(data[i]) for i in range(len(data))])
    return res


def pad_session_data_geohash(data):
    fillvalue = [0 for i in range(12)]
    # zip_longest
    data = list(zip(*itertools.zip_longest(*data, fillvalue=fillvalue)))
    res = []
    res.extend([list(data[i]) for i in range(len(data))])
    return res


def collate_session_based(batch):
    '''
    get the output of dataset.__getitem__, and perform padding
    :param batch:
    :return:
    '''
    device = batch[0][-1]
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

    X_all_loc = [item[0] for item in batch]

    X_all_loc_category = [item[7] for item in batch]
    X_all_lat = [item[8] for item in batch]
    X_all_lon = [item[9] for item in batch]
    X_all_geohash = [item[13] for item in batch]

    X_all_tim = [item[1] for item in batch]
    X_all_text = [item[2] for item in batch]
    Y_location = [lid for item in batch for lid in item[3]]

    Y_location_category = [lid for item in batch for lid in item[10]]
    Y_lat = [lid for item in batch for lid in item[11]]
    Y_lon = [lid for item in batch for lid in item[12]]
    Y_geohash = [lid for item in batch for lid in item[14]]

    target_lengths = [item[4] for item in batch]
    X_lengths = [item[5] for item in batch]
    X_users = [item[6] for item in batch]

    X_tau = torch.stack([item[15] for item in batch])
    torch.where(X_tau<5, torch.mean(X_tau), X_tau)
    Y_tau = torch.stack([item[16] for item in batch])
    torch.where(Y_tau<5, torch.mean(Y_tau), Y_tau)

    padded_X_all_loc = pad_session_data_one(X_all_loc)
    padded_X_all_loc_category = pad_session_data_one(X_all_loc_category)
    padded_X_all_lat = pad_session_data_one(X_all_lat)
    padded_X_all_lon = pad_session_data_one(X_all_lon)
    padded_X_all_geohash = pad_session_data_geohash(X_all_geohash)

    padded_X_all_tim = pad_session_data_one(X_all_tim)
    padded_X_all_loc = torch.tensor(padded_X_all_loc).long()
    padded_X_all_tim = torch.tensor(padded_X_all_tim).long()

    return session_Batch(padded_X_all_loc, padded_X_all_tim, X_all_text, Y_location, target_lengths, X_lengths, X_users,
                         padded_X_all_loc_category, padded_X_all_lat, padded_X_all_lon, Y_location_category, Y_lat,
                         Y_lon, padded_X_all_geohash, Y_geohash, X_tau, Y_tau, device)


class session_Batch():
    def __init__(self, padded_X_all_loc, padded_X_all_tim, X_all_text, Y_location, target_lengths, X_lengths, X_users,
                 padded_X_all_loc_category, padded_X_all_lat, padded_X_all_lon, Y_location_category, Y_lat, Y_lon,
                 padded_X_all_geohash, Y_geohash, X_tau, Y_tau, device):
        self.X_all_loc = torch.LongTensor(padded_X_all_loc).to(device)  # (batch, max_all_length)
        self.X_all_tim = torch.LongTensor(padded_X_all_tim).to(device)  # (batch, max_all_length)
        self.X_all_text = X_all_text
        self.X_all_loc_category = padded_X_all_loc_category
        self.X_all_lat = padded_X_all_lat
        self.X_all_lon = padded_X_all_lon
        self.X_all_geohash = torch.FloatTensor(padded_X_all_geohash).to(device)
        self.Y_location = torch.Tensor(Y_location).long().to(device)  # (Batch,) 
        self.Y_location_category = Y_location_category
        self.Y_lat = Y_lat
        self.Y_lon = Y_lon
        self.Y_geohash = Y_geohash
        self.target_lengths = target_lengths
        self.X_lengths = X_lengths
        self.X_users = torch.Tensor(X_users).long().to(device)

        self.X_tau = X_tau
        self.Y_tau = Y_tau


def pad_1d(inputs, pad=0):
    def pad_data(x, length, pad):
        x_padded = np.pad(
            x, (0, length - len(x)), mode="constant", constant_values=pad
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, pad) for x in inputs])
    return padded
