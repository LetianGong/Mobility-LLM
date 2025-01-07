import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from utils import DotDict
from model.utils import *
import torch.nn.functional as F
from torch import nn
#import faiss
from transformers import  AutoModelForCausalLM,AutoTokenizer
from .llm import LLMModel

class MobilityLLM_ModelConfig(DotDict):
    '''
    configuration of the MobilityLLM
    '''

    def __init__(self, loc_size=None, tim_size=None, uid_size=None, geohash_size=None, category_size=None, tim_emb_size=None, loc_emb_size=None,
                 hidden_size=None, user_emb_size=None, model_class=None, device=None,
                 loc_noise_mean=None, loc_noise_sigma=None, tim_noise_mean=None, tim_noise_sigma=None,
                 user_noise_mean=None, user_noise_sigma=None, tau=None,
                 pos_eps=None, neg_eps=None, dropout_rate_1=None, dropout_rate_2=None, category_vector=None, rnn_type='BiLSTM',
                 num_layers=3, k=8, momentum=0.95, temperature=0.1, theta=0.18,
                 n_components=4, shift_init=0.0, scale_init=0.0, min_clip=-5., max_clip=3., hypernet_hidden_sizes=None, max_delta_mins=1440,
                 downstream='POI',tpp='pdf',loss='pdf', dropout_spatial = None, epsilon = None, learnable_param_size=1):
        super().__init__()
        self.max_delta_mins = max_delta_mins

        self.loc_size = loc_size  #
        self.uid_size = uid_size  # 
        self.tim_size = tim_size  # 
        self.geohash_size = geohash_size  # 
        self.category_size = category_size  # 
        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size
        self.user_emb_size = user_emb_size
        self.hidden_size = hidden_size  # RNN hidden_size
        self.model_class = model_class
        self.device = device
        self.rnn_type = rnn_type
        self.num_layers = num_layers

        self.loc_noise_mean = loc_noise_mean
        self.loc_noise_sigma = loc_noise_sigma
        self.tim_noise_mean = tim_noise_mean
        self.tim_noise_sigma = tim_noise_sigma
        self.user_noise_mean = user_noise_mean
        self.user_noise_sigma = user_noise_sigma
        self.tau = tau
        self.pos_eps = pos_eps
        self.neg_eps = neg_eps
        self.dropout_rate_1 = dropout_rate_1
        self.dropout_rate_2 = dropout_rate_2
        self.downstream = downstream
        self.category_vector = category_vector
        self.learnable_param_size=learnable_param_size

        self.k = k
        self.momentum = momentum
        self.theta = theta
        self.temperature = temperature

        self.n_components = n_components    #需要
        self.min_clip = min_clip            #需要 无输入
        self.max_clip = max_clip            #需要 无输入
        self.shift_init = shift_init
        self.scale_init = scale_init
        self.hypernet_hidden_sizes = hypernet_hidden_sizes      #需要 无输入
        self.decoder_input_size = user_emb_size + hidden_size * 2    #需要
        self.loss = loss
        self.tpp = tpp
        self.dropout_spatial = dropout_spatial
        self.epsilon = epsilon

class MobilityLLM(nn.Module):
    def __init__(self, config):
        super(MobilityLLM, self).__init__()
        # initialize parameters
        self.max_delta_mins = config['max_delta_mins']
        self.truth_Y_tau = None
        self.loc_size = config['loc_size']
        self.loc_emb_size = config['loc_emb_size']
        self.tim_size = config['tim_size']
        self.tim_emb_size = config['tim_emb_size']
        self.user_size = config['uid_size']
        self.user_emb_size = config['user_emb_size']

        self.category_size = config['category_size']
        self.geohash_size = config['geohash_size']
        self.category_vector = config['category_vector']
        self.learnable_param_size = config['learnable_param_size'] # 可学习参数个数

        self.hidden_size = config['hidden_size']
        self.rnn_type = config['rnn_type']
        self.num_layers = config['num_layers']
        self.device = config['device']
        self.model_class = config['model_class']
        self.downstream = config['downstream']

        # parameters for cluster contrastive learning
        self.k = config['k']

        # parameters for time contrastive learning (Angle & Momentum based)
        # momentum
        self.momentum = config['momentum']
        # angle
        self.theta = config['theta']
        # self.theta = 0.05
        self.temperature = config['temperature']

        # spatial 
        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None
        self.softmax = nn.Softmax()
        self.epsilon = config['epsilon']
        self.sinkhorn_iterations = 3
        self.crops_for_assign = [0, 1]
        self.nmb_crops = [2]
        self.world_size = -1
        self.dropout = nn.Dropout(0.3)
        # todo 这里的0.1用超参调
        self.l2norm = True
        # location all size (embedding + geohash + category)
        self.rnn_input_size = self.loc_emb_size + self.geohash_size
        if self.rnn_type == 'BiLSTM':
            self.bi = 2
        else:
            self.bi = 1

        # parameters for social contrastive learning (4 group of parameters)
        self.para0 = nn.Parameter(torch.randn(1, 6))
        self.para1 = nn.Parameter(torch.randn(1, 4))
        self.para2 = nn.Parameter(torch.randn(1, 24))
        self.para3 = nn.Parameter(torch.randn(1, 16))

        # parameters for TPP
        self.shift_init = config['shift_init']
        self.scale_init = config['scale_init']
        self.min_clip = config['min_clip']
        self.max_clip = config['max_clip']

        ##############################################
        self.loc_noise_mean = config['loc_noise_mean']
        self.loc_noise_sigma = config['loc_noise_sigma']
        self.tim_noise_mean = config['tim_noise_mean']
        self.tim_noise_sigma = config['tim_noise_sigma']
        self.user_noise_mean = config['user_noise_mean']
        self.user_noise_sigma = config['user_noise_sigma']

        self.tau = config['tau']
        self.pos_eps = config['pos_eps']
        self.neg_eps = config['neg_eps']
        self.dropout_rate_1 = config['dropout_rate_1']
        self.dropout_rate_2 = config['dropout_rate_2']

        self.dropout_1 = nn.Dropout(self.dropout_rate_1)
        self.dropout_2 = nn.Dropout(self.dropout_rate_2)
        ################################################
        self.tpp = config['tpp']
        self.loss = config['loss']
        self.mae = torch.nn.L1Loss()
        # Embedding layer
        self.emb_loc = nn.Embedding(num_embeddings=self.loc_size, embedding_dim=self.loc_emb_size)
        self.emb_tim = nn.Embedding(num_embeddings=self.tim_size, embedding_dim=self.tim_emb_size)
        self.emb_user = nn.Embedding(num_embeddings=self.user_size, embedding_dim=self.user_emb_size)

        # Category dense layer
        self.category_dense = nn.Linear(768, self.category_size)
        # Geohash dense layer
        self.geohash_dense = nn.Linear(12, self.geohash_size)

        # rnn layer
        self.spatial_encoder = LLMModel(model_path = "/data/ZhangXinyue/MobilityLLM/params/"+ self.model_class, model_class= self.model_class, loc_size = self.loc_size, learnable_param_size = self.learnable_param_size, device = self.device)
        # if self.rnn_type == 'GRU':
        #     self.spatial_encoder = nn.GRU(self.rnn_input_size, self.hidden_size, num_layers=self.num_layers,
        #                        batch_first=False)
        #     self.temporal_encoder = nn.GRU(self.tim_emb_size + 1, self.hidden_size, num_layers=self.num_layers,
        #                        batch_first=False)
        #     self.temporal_encoder_momentum = nn.GRU(self.tim_emb_size + 1, self.hidden_size, num_layers=self.num_layers,
        #                        batch_first=False)
        # elif self.rnn_type == 'LSTM':
        #     self.spatial_encoder = nn.LSTM(self.rnn_input_size, self.hidden_size, num_layers=self.num_layers,
        #                         batch_first=False)
        # elif self.rnn_type == 'BiLSTM':
        #     self.spatial_encoder = nn.LSTM(self.rnn_input_size, self.hidden_size, num_layers=self.num_layers,
        #                         batch_first=False, bidirectional=True)
        # else:
        #     raise ValueError("rnn_type should be ['GRU', 'LSTM', 'BiLSTM']")

        #spatial_adv
        # prototype layer
        self.prototypes = None
        if isinstance(self.k, list):
            self.prototypes = MultiPrototypes(self.hidden_size, self.k)
        elif self.k > 0:
            self.prototypes = nn.Linear(self.hidden_size, self.k, bias=False)

        # projection head
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, self.hidden_size),
        )

        # Hypernet module for TPP
        self.hypernet = Hypernet(config, hidden_sizes=config.hypernet_hidden_sizes,
                                 param_sizes=[config.n_components, config.n_components, config.n_components])
        # linear for TPP
        self.linear_p1 = nn.Linear(self.hidden_size + self.user_emb_size,1024)
        self.linear_p2 = nn.Linear(1024 , 4)
        self.linear_m1 = nn.Linear(self.hidden_size + self.user_emb_size,986)
        self.linear_m2 = nn.Linear(986 , 4)
        self.linear_l1 = nn.Linear(self.hidden_size + self.user_emb_size,1560)
        self.linear_l2 = nn.Linear(1560 , 4)
        self.Tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # dense layer
        self.s2st_projection = nn.Linear(self.hidden_size * self.bi, self.hidden_size * self.bi)
        if self.downstream == 'TUL':
            self.dense = nn.Linear(in_features=self.hidden_size * self.bi, out_features=self.user_size)
            self.t2st_projection = nn.Linear(self.hidden_size * self.bi, self.hidden_size * self.bi)
            self.projection = nn.Sequential(nn.Linear(self.hidden_size * self.bi, self.hidden_size * self.bi), nn.ReLU())
        elif self.downstream == 'POI':
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size * self.bi + self.user_emb_size, self.hidden_size * self.bi + self.user_emb_size),
                nn.ReLU())
            self.dense = nn.Linear(in_features=self.hidden_size * self.bi + self.user_emb_size, out_features=self.loc_size)
            self.t2st_projection = nn.Linear(self.hidden_size * self.bi + self.user_emb_size, self.hidden_size * self.bi)
        elif self.downstream == 'TPP':
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size * self.bi + self.user_emb_size,
                          self.hidden_size * self.bi + self.user_emb_size),
                nn.ReLU())
            self.t2st_projection = nn.Linear(self.hidden_size * self.bi + self.user_emb_size,
                                             self.hidden_size * self.bi)
            final_in_size = self.hidden_size * self.bi + self.user_emb_size
            self.dense_s = nn.Linear(in_features=self.hidden_size,
                                   out_features=self.hidden_size)
            self.dense_t = nn.Linear(in_features=self.hidden_size  * self.bi + self.user_emb_size,
                                   out_features=self.hidden_size  * self.bi + self.user_emb_size)
            self.dense_st = nn.Linear(in_features=self.hidden_size  * self.bi + self.user_emb_size,
                                   out_features=self.hidden_size  * self.bi + self.user_emb_size)
            self.dense = nn.Sequential(
                nn.Linear(in_features=final_in_size, out_features=final_in_size // 4),
                nn.LeakyReLU(),
                nn.Linear(in_features=final_in_size // 4, out_features=final_in_size // 16),
                nn.LeakyReLU(),
                nn.Linear(in_features=final_in_size // 16, out_features=1),
            )
            self.linear1 = nn.Linear(self.hidden_size + self.user_emb_size, (self.hidden_size + self.user_emb_size) // 4)
            self.linear2 = nn.Linear((self.hidden_size + self.user_emb_size) // 4,
                                     (self.hidden_size + self.user_emb_size) // 16)
            self.linear3 = nn.Linear((self.hidden_size + self.user_emb_size) // 16, 1)
            self.linear0 = nn.Linear((self.hidden_size) * 2, self.hidden_size)
        else:
            raise ValueError('downstream should in [TUL, POI, TPP]!')

        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)

    def spatial_encode(self, x, time, category, geohash_, all_len, cur_len, batch_size, momentum=False, downstream='POI'):
        if momentum == True:
            f_encoder = self.spatial_encoder_momentum
        else:
            f_encoder = self.spatial_encoder
        # self-attention (mask)
        spatial_out = self.spatial_encoder(x, torch.tensor(all_len).to(self.device), time, category, geohash_) # +time, category
        # if self.rnn_type == 'GRU':
        #     spatial_out, h_n = f_encoder(packed_stuff)  # max_len*batch*hidden_size
        # elif self.rnn_type == 'LSTM':
        #     spatial_out, (h_n, c_n) = f_encoder(packed_stuff)  # max_len*batch*hidden_size
        # elif self.rnn_type == 'BiLSTM':
        #     spatial_out, (h_n, c_n) = f_encoder(packed_stuff)  # max_len*batch*hidden_size
        # else :
        #     raise ValueError('rnn type is not in GRU, LSTM, BiLSTM! ')

        # # unpack
        # spatial_out, out_len = pad_packed_sequence(spatial_out, batch_first=False)
        # spatial_out = spatial_out.permute(1, 0, 2)

        # out_len即all_len batch*max_len*hidden_size
        # concatenate
        if downstream == 'POI':
            final_out = spatial_out[0, (all_len[0] - cur_len[0]): all_len[0], :]
            for i in range(1, batch_size):
                final_out = torch.cat([final_out, spatial_out[i, (all_len[i] - cur_len[i]): all_len[i], :]], dim=0)
            # No longer concate user embedding
            # final_out = torch.cat([final_out, all_user_emb], 1)
        elif downstream == 'TPP':
            if all_len[0] == cur_len[0]:
                left = all_len[0] - cur_len[0]
                right = all_len[0]
            else:
                left = all_len[0] - cur_len[0] - 1
                right = all_len[0] - 1
            final_out = spatial_out[0, left: right, :]
            for i in range(1, batch_size):
                if all_len[i] == cur_len[i]:
                    left = all_len[i] - cur_len[i]
                    right = all_len[i]
                else:
                    left = all_len[i] - cur_len[i] - 1
                    right = all_len[i] - 1
                final_out = torch.cat([final_out, spatial_out[i, left: right, :]], dim=0)
        elif downstream == 'TUL':
            final_out = spatial_out[0, (all_len[0] - 1): all_len[0], :]
            slice_tensor = torch.mean(spatial_out[0, : all_len[0], :],dim=0,keepdim=True)
            #print("final_out_shape:",final_out.shape)
            #print("slice_tensor_shape:",slice_tensor.shape)
            for i in range(1, batch_size):
                final_out = torch.cat([final_out, torch.mean(spatial_out[i, : all_len[i], :],dim=0,keepdim=True)], dim=0)
        else:
            raise ValueError('downstream is not in [POI, TUL, TPP]')
        return final_out

    '''
    def temporal_encode(self, packed_stuff, all_len, cur_len, batch_size, momentum=False, downstream='POI'):
        if momentum == True:
            f_encoder = self.temporal_encoder_momentum
        else:
            f_encoder = self.temporal_encoder
        if self.rnn_type == 'GRU':
            temporal_out, h_n = f_encoder(packed_stuff)  # max_len*batch*hidden_size
        elif self.rnn_type == 'LSTM':
            temporal_out, (h_n, c_n) = f_encoder(packed_stuff)  # max_len*batch*hidden_size
        elif self.rnn_type == 'BiLSTM':
            temporal_out, (h_n, c_n) = f_encoder(packed_stuff)  # max_len*batch*hidden_size
        else :
            raise ValueError('rnn type is not in GRU, LSTM, BiLSTM! ')

        # unpack
        temporal_out, out_len = pad_packed_sequence(temporal_out, batch_first=False)
        temporal_out = temporal_out.permute(1, 0, 2)

        # out_len即all_len batch*max_len*hidden_size
        # concatenate
        if downstream == 'POI':
            final_out = temporal_out[0, (all_len[0] - cur_len[0]): all_len[0], :]
            for i in range(1, batch_size):
                final_out = torch.cat([final_out, temporal_out[i, (all_len[i] - cur_len[i]): all_len[i], :]], dim=0)

        elif downstream == 'TPP':
            if all_len[0] == cur_len[0]:
                left = all_len[0] - cur_len[0]
                right = all_len[0]
            else:
                left = all_len[0] - cur_len[0] - 1
                right = all_len[0] - 1
            final_out = temporal_out[0, left: right, :]
            for i in range(1, batch_size):
                if all_len[i] == cur_len[i]:
                    left = all_len[i] - cur_len[i]
                    right = all_len[i]
                else:
                    left = all_len[i] - cur_len[i] - 1
                    right = all_len[i] - 1
                final_out = torch.cat([final_out, temporal_out[i, left: right, :]], dim=0)
        elif downstream == 'TUL':
            final_out = temporal_out[0, (all_len[0] - 1): all_len[0], :]
            for i in range(1, batch_size):
                final_out = torch.cat([final_out, temporal_out[i, (all_len[i] - 1): all_len[i], :]], dim=0)
        else:
            raise ValueError('downstream is not in [POI, TUL, TPP]')

        return final_out
    '''


    def normal_logpdf(self, x, mean, log_scale):
        '''
        log pdf of the normal distribution with mean and log_sigma
        '''
        # z = (x - mean[:,0:1]) * torch.exp(-log_scale[:,0:1])
        z = (x - mean) * torch.exp(-log_scale)
        return -log_scale - 0.5 * z.pow(2.0) - 0.5 * np.log(2 * np.pi)

    def mixnormal_logpdf(self, x, log_prior, means, log_scales):
        '''
        :param x: ground truth
        :param log_prior:  归一化后的权重系数，(batch,max_length/actual_length,n_components)=(64, 128, 64),
        :param means: (batch,max_length/actual_length,n_components)=(64, 128, 64)
        :param log_scales: (batch,max_length/actual_length,n_components)=(64, 128, 64), scales对应论文中的s
        :return:
        '''
        return torch.logsumexp(
            log_prior + self.normal_logpdf(x, means, log_scales),
            dim=-1
        )

    def get_params(self, decoder_input):
        """
        Generate model parameters based on the inputs
        Args:
            input: decoder input [batch, decoder_input_size]

        Returns:
            prior_logits: shape [batch, n_components]
            means: shape [batch, n_components]
            log_scales: shape [batch, n_components]
        """
        prior_logits, means, log_scales = self.hypernet(decoder_input)

        # Clamp values that go through exp for numerical stability
        prior_logits = clamp_preserve_gradients(prior_logits, self.min_clip, self.max_clip)
        log_scales = clamp_preserve_gradients(log_scales, self.min_clip, self.max_clip)

        # normalize prior_logits
        prior_logits = F.log_softmax(prior_logits, dim=-1)  # 这里进行了权重w的归一化,而且之后进行了log
        return prior_logits, means, log_scales



    def forward(self, batch, mode='test', cont_conf=None, downstream='POI', queue = None, use_the_queue = False):


        loc = batch.X_all_loc


        tim = batch.X_all_tim
        user = batch.X_users
        geohash_ = batch.X_all_geohash
        cur_len = batch.target_lengths
        all_len = batch.X_lengths
        loc_cat = batch.X_all_loc_category

        batch_size = loc.shape[0]
        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        user_emb = self.emb_user(user)
        geohash_ = self.geohash_dense(geohash_)

        # concatenate
        #x = torch.cat([loc_emb, tim_emb], dim=2)
        x = loc.to(self.device)
        time = tim.float().to(self.device)
        # x = torch.cat([loc_emb, geohash_], dim=2).permute(1, 0, 2)
        #
        # x_temporal = tim_emb.permute(1, 0, 2)

        # cat locs & taus
        all_tau = [torch.cat((batch.X_tau[0, :all_len[0] - cur_len[0]], batch.Y_tau[0, :cur_len[0]]), dim=-1)]
        self.truth_Y_tau = all_tau[0][all_len[0] - cur_len[0]:all_len[0]]

        for i in range(1, batch_size):
            # taus
            cur_tau = torch.cat((batch.X_tau[i, :all_len[i] - cur_len[i]], batch.Y_tau[i, :cur_len[i]]), dim=-1)
            all_tau.append(cur_tau)

            self.truth_Y_tau = torch.cat((self.truth_Y_tau, all_tau[i][all_len[i] - cur_len[i]:all_len[i]]), dim=0)

        all_tau = pad_sequence(all_tau, batch_first=False).to(self.device)
        # x_temporal = torch.cat((all_tau.unsqueeze(-1), x_temporal), dim=-1)

        # # pack
        # pack_x = pack_padded_sequence(x, lengths=all_len, enforce_sorted=False)
        # pack_x_temporal = pack_padded_sequence(x_temporal, lengths=all_len, enforce_sorted=False)

        final_out = self.spatial_encode(x, all_tau.transpose(0,1), loc_cat, geohash_, all_len, cur_len, batch_size, downstream=downstream)
        # final_temporal_out = self.temporal_encode(pack_x_temporal, all_len, cur_len, batch_size, downstream=downstream)

        all_user_emb = user_emb[0].unsqueeze(dim=0).repeat(cur_len[0], 1)
        for i in range(1, batch_size):
            all_user_emb = torch.cat([all_user_emb, user_emb[i].unsqueeze(dim=0).repeat(cur_len[i], 1)], dim=0)
        
        #todo 去掉时间，试一下结果
        if downstream == 'POI':
            prediction_out = torch.cat([final_out, all_user_emb], 1)
            dense = self.dense(prediction_out)  # Batch * loc_size
            pred = nn.LogSoftmax(dim=1)(dense)  # result
        elif downstream == 'TUL':
            # prediction_out = torch.cat([final_spatial_out, final_temporal_out], 1)
            dense = self.dense(self.dropout(final_out))
            pred = nn.LogSoftmax(dim=1)(dense)  # result
        elif downstream == 'TPP':
            final_out = torch.cat([final_out, all_user_emb], 1)
            prediction_out = self.dense(final_out)  # Batch * loc_size
        else:
            raise ValueError('downstream is not in [POI, TUL, TPP]')


        criterion = nn.NLLLoss().to(self.device)
        criterion1 = nn.L1Loss().to(self.device)


        if downstream == 'POI':
            s_loss_score = criterion(pred, batch.Y_location).requires_grad_(True)
            _, top_k_pred = torch.topk(pred, k=self.loc_size)  # (batch, K)=(batch, num_class)
        elif downstream == 'TUL':
            s_loss_score = criterion(pred, batch.X_users).requires_grad_(True)
            _, top_k_pred = torch.topk(pred, k=self.user_size)  # (batch, K)=(batch, num_class)
        elif downstream == 'TPP':
            # y = torch.log(self.truth_Y_tau + 1e-2).unsqueeze(-1)

            mean_time = self.linear3(self.sigmoid(self.linear2(self.sigmoid(self.linear1(final_out)))))
            # loss = self.mae(mean_time, y.to(mean_time.device))

            # mean_time = torch.exp(mean_time)
            # y = torch.log(self.truth_Y_tau + 1e-2).unsqueeze(-1)
            # y = (y - self.shift_init.to(y.device)) / self.scale_init.to(y.device)
            # if self.tpp == 'pdf':
            #     prior_logits, means, log_scales = self.get_params(prediction_out)
            # #todo 用可学习参数*pred_out代替以上三个
            # elif self.tpp == 'linear':
            #     prior_logits = self.linear_p2(self.Tanh(self.linear_p1(prediction_out)))
            #     means = self.linear_m2(self.Tanh(self.linear_m1(prediction_out)))
            #     log_scales = self.linear_l2(self.Tanh(self.linear_l1(prediction_out)))
            # # 使用线性层替代
            # log_p = self.mixnormal_logpdf(y.to(self.device), prior_logits, means, log_scales)
            # prior = prior_logits.exp()  # (batch, n_components=64)
            # scales_squared = (log_scales * 2).exp()
            # a = self.scale_init.to(y.device)
            # b = self.shift_init.to(y.device)
            # mean_time = (prior * torch.exp(a * means + b + 0.5 * a ** 2 * scales_squared)).sum(-1)
            # mean_time = torch.clip(mean_time, max=float(self.max_delta_mins), min=0.0)
            # if self.loss == 'pdf':
            #     s_loss_score = -log_p.mean()
            if self.loss == 'mae':
                # s_loss_score = criterion1(prediction_out.to(y.device),(self.truth_Y_tau).to(y.device))
                s_loss_score = criterion1(prediction_out.squeeze().to('cpu'),self.truth_Y_tau.to('cpu'))
            top_k_pred = prediction_out
        else:
            raise ValueError('downstream is not in [POI, TUL, TPP]')

        if mode == 'train' and sum(cont_conf) != 0:
            return s_loss_score, top_k_pred, queue
        else:
            return s_loss_score, top_k_pred, queue

class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out