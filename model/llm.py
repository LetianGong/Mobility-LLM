import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from peft import LoraModel, LoraConfig

from .base import *
from .bert import get_batch_mask
from .phi_model import PhiModel
from .positional_encoding import LearnableFourierFeatures as LFF


def get_encoder(model_path, model_class):
    if model_class == 'tinybert':
        lora_config = LoraConfig(
            task_type="SEQ_2_SEQ_LM",
            r=8,  # Lora attention dimension.
            lora_alpha=32,  # The alpha parameter for Lora scaling.
            target_modules=["query", "value"],  # The names of the modules to apply Lora to.
            lora_dropout=0.01,  # The dropout probability for Lora layers.
        )
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        emb_size = model.config.emb_size
        hidden_size = model.config.hidden_size

    elif model_class == 'phi-2':
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,  # Lora attention dimension.
            lora_alpha=32,  # The alpha parameter for Lora scaling.
            target_modules=["Wqkv"],  # The names of the modules to apply Lora to.
            lora_dropout=0.01,  # The dropout probability for Lora layers.
        )
        model = CustomPhiModel.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            # device_map="cuda",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        emb_size = model.config.n_embd
        hidden_size = model.config.n_embd

    elif model_class == 'gpt2':
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,  # Lora attention dimension.
            lora_alpha=32,  # The alpha parameter for Lora scaling.
            target_modules=["c_attn"],  # The names of the modules to apply Lora to.
            lora_dropout=0.02,  # The dropout probability for Lora layers.
        )
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        emb_size = model.config.n_embd
        hidden_size = model.config.n_embd
     
    elif model_class == 'TinyLlama-1_1B':
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,  # Lora attention dimension.
            lora_alpha=32,  # The alpha parameter for Lora scaling.
            target_modules=["q_proj","k_proj"],  # The names of the modules to apply Lora to.
            lora_dropout=0.02,  # The dropout probability for Lora layers.
        )
        model = CustomLlamaModel(model_path='params/TinyLlama-1.1B')
        tokenizer = AutoTokenizer.from_pretrained('params/TinyLlama-1.1B')
        emb_size = 2048
        hidden_size = 2048
      
    elif model_class == 'TinyLlama-Chat':
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,  # Lora attention dimension.
            lora_alpha=32,  # The alpha parameter for Lora scaling.
            target_modules=["q_proj","k_proj"],  # The names of the modules to apply Lora to.
            lora_dropout=0.02,  # The dropout probability for Lora layers.
        )
        model = CustomLlamaModel(model_path= 'params/TinyLlama-Chat')
        tokenizer = AutoTokenizer.from_pretrained('params/TinyLlama-Chat')
        emb_size = 2048
        hidden_size = 2048
      
    elif model_class == 'pythia-70M':
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,  # Lora attention dimension.
            lora_alpha=32,  # The alpha parameter for Lora scaling.
            target_modules=["query_key_value"],  # The names of the modules to apply Lora to.
            lora_dropout=0.02,  # The dropout probability for Lora layers.
        )
        model = CustomPythiaModel(model_path= 'params/pythia-70M')
        tokenizer = AutoTokenizer.from_pretrained('params/pythia-70M')
        emb_size = 512
        hidden_size = 512
       
    elif model_class == 'pythia-2_8B':
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,  # Lora attention dimension.
            lora_alpha=32,  # The alpha parameter for Lora scaling.
            target_modules=["query_key_value"],  # The names of the modules to apply Lora to.
            lora_dropout=0.02,  # The dropout probability for Lora layers.
        )
        model = CustomPythiaModel(model_path= 'params/pythia-2.8B')
        tokenizer = AutoTokenizer.from_pretrained('params/pythia-2.8B')
        emb_size = 2560
        hidden_size = 2560
       
    elif model_class == 'pythia-1B':
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,  # Lora attention dimension.
            lora_alpha=32,  # The alpha parameter for Lora scaling.
            target_modules=["query_key_value"],  # The names of the modules to apply Lora to.
            lora_dropout=0.02,  # The dropout probability for Lora layers.
        )
        model = CustomPythiaModel(model_path= 'params/pythia-1B')
        tokenizer = AutoTokenizer.from_pretrained('params/pythia-1B')
        emb_size = 2048
        hidden_size = 2048
        
    elif model_class == 'LiteLlama':
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,  # Lora attention dimension.
            lora_alpha=32,  # The alpha parameter for Lora scaling.
            target_modules=["q_proj","k_proj"],  # The names of the modules to apply Lora to.
            lora_dropout=0.02,  # The dropout probability for Lora layers.
        )
        model = CustomLlamaModel(model_path= 'params/LiteLlama')
        tokenizer = AutoTokenizer.from_pretrained('params/LiteLlama')
        emb_size = 1024
        hidden_size = 1024

    else:
        raise NotImplementedError("model_class should be one of ['tinybert', 'phi-2']")

    return LoraModel(model, lora_config, model_class), tokenizer, emb_size, hidden_size
    # return model, emb_size, hidden_size


class CustomPhiModel(PhiModel):
    """ Phi for traj modeling """

    _keys_to_ignore_on_load_missing = ["ladder_side_nets", "up_net"]
    # _keys_to_ignore_on_load_unexpected = [r"transformer\.h\.\d+\.mlp.(fc_in|fc_out)\.(weight|bias)"]

    def __init__(self, config, r=32):
        super().__init__(config)

        assert config.n_embd % r == 0, f"n_embd should be divisible by r, got {config.n_embd} and {r}"
        side_dim = config.n_embd // r

        self.side_dim = side_dim
        self.ladder_side_nets = nn.ModuleList([nn.Linear(config.n_embd, side_dim) for _ in range(config.n_layer)])
        self.up_net = nn.Linear(side_dim, config.n_embd)

    def forward(self, input_ids=None, inputs_embeds=None, past_key_values=None, attention_mask=None):
        if inputs_embeds is None:
            assert input_ids is not None, "You have to specify either input_ids or inputs_embeds"
            hidden_states = self.embd(input_ids)
        else:
            hidden_states = inputs_embeds

        #hidden_states_backbone = hidden_states.detach()
        #side_states = torch.zeros(*hidden_states.shape[:-1], self.side_dim,
        #                          dtype=hidden_states.dtype).to(hidden_states.device)
        for i, layer in enumerate(self.h):
           hidden_states = layer(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            )
            
        #hidden_states = self.up_net(side_states) + hidden_states
        return hidden_states

class CustomLlamaModel(nn.Module):
    """ Phi for traj modeling """


    def __init__(self, model_path):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(model_path).model
        self.embd = self.model.embed_tokens 

    def forward(self, input_ids=None, inputs_embeds=None, past_key_values=None, attention_mask=None):
        past_key_values_length = 0
        
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(0, inputs_embeds.shape[1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)       
      
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () 
        
      
        for decoder_layer in self.model.layers:
            
            all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,               
                
            )

            hidden_states = layer_outputs[0]            

        hidden_states = self.model.norm(hidden_states)
        # add hidden states from the last decoder layer        
        all_hidden_states += (hidden_states,)
        return all_hidden_states

class CustomPythiaModel(nn.Module):
    """ Phi for traj modeling """


    def __init__(self, model_path):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(model_path).gpt_neox
        self.embd =  self.model.embed_in
        # 模型中有self.emb_dropout，是否也要加？
        # forword函数相关代码为：
        # if inputs_embeds is None:
        #     inputs_embeds = self.embed_in(input_ids)
        # hidden_states = self.emb_dropout(inputs_embeds)


    def forward(self, input_ids=None, inputs_embeds=None, past_key_values=None, attention_mask=None):
        past_key_values_length = 0
        
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(0, inputs_embeds.shape[1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)       
      
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () 
        past_length = 0
        past_key_values = tuple([None] * self.model.config.num_hidden_layers)
      
        for i, (layer, layer_past) in enumerate(zip(self.model.layers, past_key_values)):
            
            all_hidden_states = all_hidden_states + (hidden_states,)

            
            
            outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                
                layer_past=layer_past,
               
            )
            hidden_states = outputs[0]
           
        hidden_states = self.model.final_layer_norm(hidden_states)
        # Add last hidden state
      
        all_hidden_states = all_hidden_states + (hidden_states,)
        return all_hidden_states
        
class LLMModel(Encoder):
    def __init__(self, model_path,loc_size,device,output_size=256,
                 
                 model_class='gpt2', learnable_param_size=1,):
        super().__init__('LLM-gpt2')

        self.output_size = output_size
        self.loc_size=loc_size
        self.model_class = model_class
        self.device = device
        self.learnable_param_size=learnable_param_size

        self.encoder, self.tokenizer, self.emb_size, self.hidden_size = get_encoder(model_path, model_class)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Froze the parameters.
        for i, (name, param) in enumerate(self.encoder.named_parameters()):
            #param.requires_grad = False

            # if 'side_net' in name or 'up_net' in name:
                # param.requires_grad = True
            # else:
            #     param.requires_grad = False

            # if 'ln' in name or 'wpe' in name:  # or 'mlp' in name:
            #     param.requires_grad = True
            # elif 'mlp' in name:
            #     param.requires_grad = False
            # else:
            #     param.requires_grad = False
            print(name, param.requires_grad)

        self.embedder = nn.Embedding(self.loc_size,self.emb_size)
        self.out_linear = nn.Sequential(nn.Linear(self.hidden_size, output_size, bias=False),
                                        nn.LayerNorm(output_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(output_size, output_size))
        # # f_dim,h_dim的取值问题（暂时）
        # self.lff = LFF(pos_dim=1, f_dim=128, h_dim=256, d_dim=self.emb_size) # learnable fourier features module
        self.time_linear = nn.Sequential(nn.Linear(1, 16),nn.LayerNorm(16),
                                        nn.ReLU(inplace=True),nn.Linear(16, 16))
        self.cat_linear = nn.Sequential(nn.Linear(self.emb_size+16, self.emb_size),nn.LayerNorm(self.emb_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.emb_size, self.emb_size))                 
                                      
        # TEMPO
        self.cls_token = nn.Parameter(torch.zeros(self.learnable_param_size, self.emb_size).float(), requires_grad=True)

    def forward(self, x, valid_len, time, category, geohash_, **kwargs):
        return self.forward_suffix(x, valid_len, time, category, geohash_, self.cls_token)

    def forward_suffix(self, x, valid_len, time, category, geohash_, tokens):
        """ P-tuning-like suffix forward. """
        B, L, E_in = x.unsqueeze(-1).shape  # time.shape=[B,L], category.shape=[B,L]
        
        # TEMPO改进方案
        trip_batch_mask = get_batch_mask(B, L+self.learnable_param_size, valid_len)
        batch_mask = get_batch_mask(B, L+self.learnable_param_size, valid_len+self.learnable_param_size)

        masked_values = torch.zeros_like(x)
        
        x = torch.where(get_batch_mask(B, L, valid_len).unsqueeze(-1), x.unsqueeze(-1), masked_values.unsqueeze(-1))
        x_embeddings = self.embedder(x).squeeze(-2) # (B,L,self.emb_size)
        # TEST
        '''
        # Basic usage of Learnable-Fourier-Features
        lff = LFF(pos_dim=2, f_dim=128, h_dim=256, d_dim=64) # learnable fourier features module
        pos = torch.randn([4, 1024, 1, 2])  # random positional coordinates
        pe = lff(pos)  # forward
        ''' #128*173
        # time_embeddings = self.lff(time.unsqueeze(-1).unsqueeze(-1)) # (B L G M) → (B L D)
        # x_embeddings += time_embeddings * 0.01
        ## x_embeddings = self.cat_linear(torch.cat((x_embeddings, time_embeddings), dim=-1))
        time_embeddings = self.time_linear(time.unsqueeze(-1))

        category_vocabIds = []
        for i, seq in enumerate(category): # 对batch数据应该还是得用for循环
            input_seq = seq[:valid_len[i]] # seq中的pad符号为数字0（不是str，麻烦），先获取有效长度的seq
            if len(set(input_seq))==1 and '' in input_seq:  # seq存在['','',……,0,0,0,……]的情况（最麻烦），替换''（暂）
                input_seq = [self.tokenizer.eos_token] * valid_len[i]
            # 对每个序列取category对应的vocabId的最后一个
            # 不用for循环实现，需要先对tokens做padding，再取最后一个非padding的——可能不如普通的for循环实现快
            category_vocabId = self.tokenizer(input_seq, padding=True, return_tensors="pt")['input_ids']
            last_index = torch.where(category_vocabId != self.tokenizer.eos_token_id, torch.full_like(category_vocabId, 1), 0).sum(dim=1)
            category_vocabId = [int(category_vocabId[index][int(x.item()) - 1].item()) for index, x in enumerate(last_index)]
            '''
            category_vocabId = []
            for i in seq:
                if i == '': # test时，若出现''要特殊处理一下，否则报错
                category_vocabId.append(torch.tensor(self.tokenizer.eos_token_id))
                else:
                    category_vocabId.append(self.tokenizer(i, return_tensors="pt")['input_ids'][-1][-1])
            category_vocabId = torch.tensor(category_vocabId)
            '''
            category_vocabIds.append(torch.tensor(category_vocabId+[self.tokenizer.eos_token_id]*(L-valid_len[i])))
        category_vocabIds = torch.stack(category_vocabIds,dim=0).to(self.device)
        if self.model_class == 'gpt2':
          category_embeddings = self.encoder.model.transformer.wte(category_vocabIds)
        else:
          category_embeddings = self.encoder.model.embd(category_vocabIds)
        #x_embeddings += category_embeddings
          
        #x_embeddings = self.cat_linear(torch.cat((x_embeddings, time_embeddings, category_embeddings), dim=-1))
        x_embeddings = self.cat_linear(torch.cat((x_embeddings, time_embeddings), dim=-1))

        h = torch.zeros(B, L+self.learnable_param_size, self.emb_size).to(x.device)

        h[:, :-self.learnable_param_size] = x_embeddings
        # position = (batch_mask.long() - trip_batch_mask.long()) == 1
        # temp1 = h[(batch_mask.long() - trip_batch_mask.long()) == 1]
        # temp2 = tokens.repeat(B,1)
        h[(batch_mask.long() - trip_batch_mask.long()) == 1] = tokens.repeat(B,1)
        # h[(batch_mask.long() - trip_batch_mask.long()) == 1] = token
        if self.model_class == 'tinybert':
            h = self.encoder(inputs_embeds=h, attention_mask=batch_mask.unsqueeze(-1), output_hidden_states=True).hidden_states[-1]
        elif self.model_class == 'phi-2':
            h = self.encoder(inputs_embeds=h, attention_mask=batch_mask)
        elif self.model_class == 'gpt2':
            h = self.encoder(inputs_embeds=h, attention_mask=batch_mask.unsqueeze(-1), output_hidden_states=True).hidden_states[-1]
        elif self.model_class == 'LiteLlama':
            h = self.encoder(inputs_embeds=h, attention_mask=batch_mask.unsqueeze(-1).unsqueeze(1).expand(-1,-1,-1,batch_mask.shape[-1]))[0]
        elif self.model_class == 'TinyLlama-Chat':
            h = self.encoder(inputs_embeds=h, attention_mask=batch_mask.unsqueeze(-1).unsqueeze(1).expand(-1,-1,-1,batch_mask.shape[-1]))[0]
        elif self.model_class == 'TinyLlama-1_1B':
            h = self.encoder(inputs_embeds=h, attention_mask=batch_mask.unsqueeze(-1).unsqueeze(1).expand(-1,-1,-1,batch_mask.shape[-1]))[0]
        elif self.model_class == 'pythia-70M':
            h = self.encoder(inputs_embeds=h, attention_mask=batch_mask.unsqueeze(-1).unsqueeze(1).expand(-1,-1,-1,batch_mask.shape[-1]))[0]
        elif self.model_class == 'pythia-2_8B':
            h = self.encoder(inputs_embeds=h, attention_mask=batch_mask.unsqueeze(-1).unsqueeze(1).expand(-1,-1,-1,batch_mask.shape[-1]))[0]
        elif self.model_class == 'pythia-1B':
            h = self.encoder(inputs_embeds=h, attention_mask=batch_mask.unsqueeze(-1).unsqueeze(1).expand(-1,-1,-1,batch_mask.shape[-1]))[0]
        h = torch.nan_to_num(h)
        #output = self.out_linear(h[:, -1])
        output = self.out_linear(h)

        return output