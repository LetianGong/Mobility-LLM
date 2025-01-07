# title
Mobility-LLM: Learning Visiting Intentions and Travel Preferences from Human Mobility Data with Large Language Models
# Datasets
To demonstrate the superiority of our proposed model, our experiements are carried out on four real-world datasets derived from Gowalla (GOW), WeePlace (WEE), Brightkite (BKC) and FourSquare (TKY) check-in data.

In order to facilitate the training of our model, Our model undergoes a filtering process that selects high-quality check-in sequences for training. To ensure data consistency, we set a maximum historical time limit of 120 days and filter out users with fewer than 10 records and places visited fewer than 10 times.

The below table shows the statistics of three datasets.
|             |  Gowalla  |  WeePlace  |  Brightkite  |  FourSquare  |
| ----------  |:---------:|:----------:|:------------:|:------------:|
| # users     | 5,853     | 1,028      | 431          | 703          |  
| # POIs      | 52,032    | 9,295      | 3,554        | 11,117       |
| # Samples   | 413,563   | 104,762    | 44,716       | 60,734       |

- Download processed datasets from following sources:
    - <a href='https://1drv.ms/f/c/401a59cded375360/EuI9Lfh3qhVPrMdO22wDbc0BjJ_-5M3YIPEOLyrGoEMD3A?e=jBGTCC'>One Drive</a> 
      - https://1drv.ms/f/c/401a59cded375360/EuI9Lfh3qhVPrMdO22wDbc0BjJ_-5M3YIPEOLyrGoEMD3A?e=jBGTCC
    - <a href='https://pan.baidu.com/s/1yKBDXZxnIX9r1km7gh3tyg'>BaiduNetDisk</a> code: `x24z`
      - https://pan.baidu.com/s/1nWWMzS1yQaGdaiVd-njJxQ 
  - Copy all files and directories to `MobilityLLM/data/new_datasets`

# Large Lanugage Models
We compare eight representative backbones with varying capacities, including TinyLlama, TinyLlama-Chat, LiteLlama, phi-2, pythia-70M, pythia-1B, pythia-2.8B and GPT-2.
- Download models from following sources:
  - <a href='https://huggingface.co/models'>https://huggingface.co/models</a>
  - Copy all files and directories to `MobilityLLM/params/*/`

# Requirements
- python >= 3.6
- PyTorch >= 1.8

# Usage :
  Enter directory `MobilityLLM`.
  Downstream tasks:
  Location Prediction (LP), Trajectory User Link (TUL), or Time Prediction (TP).
  model class:
  TinyLlama-1_1B (TinyLlama), TinyLlama-Chat (TinyLlama-Chat), phi-2 (phi-2), pythia-70M (pythia-70M), pythia-2_8B (pythia-2.8B), pythia-1B (pythia-1B), LiteLlama (LiteLlama), gpt2 (GPT-2).
  - Train model on WEE of LP task:
    `python train_MobilityLLM.py --config config/MobilityLLM_wee_POI.conf --dataroot data/ --model_class`
    <br>
  - Train model on TKY of LP task:
    `python train_MobilityLLM.py --config config/MobilityLLM_tky_POI.conf --dataroot data/ --model_class`
    <br>
  - Train model on GOW of LP task:
    `python train_MobilityLLM.py --config config/MobilityLLM_gow_POI.conf --dataroot data/ --model_class`
    <br>
  - Train model on BKC of LP task:
    `python train_MobilityLLM.py --config config/MobilityLLM_bkc_POI.conf --dataroot data/ --model_class`
    <br>
  - Train model on WEE of TUL task:
    `python train_MobilityLLM.py --config config/MobilityLLM_wee_TUL.conf --dataroot data/ --model_class`
    <br>
  - Train model on TKY of TUL task:
    `python train_MobilityLLM.py --config config/MobilityLLM_tky_TUL.conf --dataroot data/ --model_class`
    <br>
  - Train model on GOW of TUL task:
    `python train_MobilityLLM.py --config config/MobilityLLM_gow_TUL.conf --dataroot data/ --model_class`
    <br>
  - Train model on BKC of TUL task:
    `python train_MobilityLLM.py --config config/MobilityLLM_bkc_TUL.conf --dataroot data/ --model_class`
    <br>
  - Train model on WEE of TP task:
    `python train_MobilityLLM.py --config config/MobilityLLM_wee_TPP.conf --dataroot data/ --model_class`
    <br>
  - Train model on TKY of TP task:
    `python train_MobilityLLM.py --config config/MobilityLLM_tky_TPP.conf --dataroot data/ --model_class`
    <br>
  - Train model on GOW of TP task:
    `python train_MobilityLLM.py --config config/MobilityLLM_gow_TPP.conf --dataroot data/ --model_class`
    <br>
  - Train model on BKC of TP task:
    `python train_MobilityLLM.py --config config/MobilityLLM_bkc_TPP.conf --dataroot data/ --model_class`
# Configuration
The configuration file `MobilityLLM_*.conf` contains three parts: Data, Training and Model:

## Data
- dataset_name: The name of the datasets, represents www_GOW, www_BKC, www_TKY or www_WEE.
- max_his_period_days: The max history time.
- max_merge_seconds_limit: To judge whether two identical locations are the same event.
- max_delta_mins: To limit the prediction range.
- least_disuser_count: To filter locations, keep locations which have at least * users.
- least_checkins_count: To filter users, keep users who have at least * checkins.
- split_save: 1 or 0, representing whether datasets are split saved.

## Training
- mode: train for default, 
- ctx: cuda index, 0 for default
- regularization: float, regularization factor.
- learning_rate: float
- max_epochs: int
- display_step: int
- patience: int, for early stopping.
- train_batch: int
- val_batch: int
- test_batch: int
- batch_size: int
- save_results: bool

## Model
- adv: 0 or 1, enable adversarial or not.
- downstream: POI, TUL or TPP, representing Location Prediction, Trajectory User Link, and Time Prediction respestively.

The remaining parameters are the best parameters of the model.
