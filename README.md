# DeepRec-torch
DeepRec-torch is a framework based a pytorch. 
This project is more like a tutorial for learning recommender system models than a tool for direct using.

## Dependency

- torch 1.2.0
- numpy 1.17.3
- pandas 0.25.3
- scikit-learn 0.21.3
- tensorboard 2.2.1 (For loss and metrics visualization)

## Quick Start
1.Load and preprocess data
```python
from example.loader.criteo_loader import load_data,missing_values_process
# load 10,000 pieces from criteo-1m dataset
data = load_data('/path/to/data',n_samples=10000)
data = missing_values_process(data)
```

2.Describe the columns with FeatureMeta
```python
from feature.feature_meta import FeatureMeta
from example.loader.criteo_loader import continuous_columns,category_columns

feature_meta = FeatureMeta()
for column in continuous_columns:
    # By default, the continuous feature will not be discretized.
    feature_meta.add_continuous_feat(column)
for column in category_columns:
    feature_meta.add_categorical_feat(column)
```

3.Transform data into wanted format (usually feat_index and feat_value)
```python
from preprocess.feat_engineering import preprocess_features

x_idx, x_value = preprocess_features(feature_meta, data)

label = data.y
```

4.Prepare for training
```python
import torch

# Assign the device for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data into assigned device
X_idx_tensor_gpu = torch.LongTensor(x_idx).to(device)
X_value_tensor_gpu = torch.Tensor(x_value).to(device)
y_tensor_gpu = torch.Tensor(label).to(device)

# Note that a binary classifier requires label with shape (n_samples,1)
y_tensor_gpu = y_tensor_gpu.reshape(-1, 1)

# Form a dataset for torch`s DataLoader
X_cuda = TensorDataset(X_idx_tensor_gpu, X_value_tensor_gpu, y_tensor_gpu)
```

5.Load a model and set parameters (pre-defined models for ctr prediction task are in model.ctr package)
```python
from model.ctr.fm import FM

# Create an FM model with embedding size of 5 and binary output, and load it into the assigned device
fm_model = FM(emb_dim=5, feat_dim=feat_meta.get_num_feats(), out_type='binary').to(device)

# Assign an optimizer for the model
optimizer = torch.optim.Adam(fm_model.parameters(), lr=1e-4)
```

6.Train the model with a trainer
```python
from util.train import train_model_hold_out

# Train the model with hold-out model selection
train_model_hold_out(job_name='fm-binary-cls', device=device,
                     model=fm_model, dataset=X_cuda,
                     loss_func=nn.BCELoss(), optimizer=optimizer,
                     epochs=20, batch_size=256)
# Checkpoint saving is by default true in trainers. 
# For more custom settings, create a dict like follow:
ckpt_settings = {'save_ckpt':True, 'ckpt_dir':'path/to/ckpt_dir', 'ckpt_interval':3}
# Then send the kwargs parameter
train_model_hold_out(...,**ckpt_settings)
# Settings for log file path, model saving path and tensorboard file path is similar, see util.train.py
```
The role of the trainer is more a log writer than a simple model training method.

For more examples:

- Model usage examples are available in example.model package.

- Data loader examples are available in example.loader package.

- Dataset EDA examples are available in example.eda package with jupyter notebook format.

## Change Log

See changelog.md

## Model list
### Click Through Rate Prediction
| model | paper |
|:-----|:------|
|LR: Logistic Regression| [Simple and Scalable Response Prediction for Display Advertising][LR]|
|FM: Factorization Machine|\[ICDM 2010\][Factorization Machines][FM]|
|GBDT+LR: Gradient Boosting Tree with Logistic Regression|[Practical Lessons from Predicting Clicks on Ads at Facebook][GBDTLR]|
|FNN: Factorization-supported Neural Network|\[ECIR 2016\][Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction][FNN]|
|PNN: Product-based Neural Network|\[ICDM 2016\][Product-based neural networks for user response prediction][PNN]|
|Wide and Deep|\[DLRS 2016\][Wide & Deep Learning for Recommender Systems][WideDeep]|
|DeepFM|\[IJCAI 2017\][DeepFM: A Factorization-Machine based Neural Network for CTR Prediction][DeepFM]|
|AFM: Attentional Factorization Machine|\[IJCAI 2017\][Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks][AFM]|
|NFM: Neural Factorization Machine|\[SIGIR 2017\][Neural Factorization Machines for Sparse Predictive Analytics][NFM]|
<!--
|FFM: Field-aware Factorization Machine|\[RecSys 2016\][Field-aware Factorization Machines for CTR Prediction][FFM]|
|CCPM: Convolutional Click Prediction Model|\[CIKM 2015\][A Convolutional Click Prediction Model][CCPM]|
|Piece-wise Linear Model|\[arxiv 2017\][Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction][PLM]|
|DCN: Deep & Cross Network|\[ADKDD 2017\][Deep & Cross Network for Ad Click Predictions][DCN]|
|xDeepFM|\[KDD 2018\][xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems][xDeepFM]|
|AutoInt|\[arxiv 2018\][AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks][AutoInt]|
|ONN|\[arxiv 2019\][Operation-aware Neural Networks for User Response Prediction][ONN]|
|FGCNN|\[WWW 2019\][Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction][FGCNN]|
|FiBiNET|\[RecSys 2019\][FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction][FiBiNET]|
|FLEN|\[arxiv 2019\][FLEN: Leveraging Field for Scalable CTR Prediction][FLEN]|
 -->
[LR]:https://dl.acm.org/doi/pdf/10.1145/2532128?download=true
[FM]:https://dl.acm.org/doi/10.1109/ICDM.2010.127
[GBDTLR]:https://dl.acm.org/doi/pdf/10.1145/2648584.2648589
[CCPM]:http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf
[FFM]:https://dl.acm.org/doi/pdf/10.1145/2959100.2959134
[FNN]:https://arxiv.org/pdf/1601.02376.pdf
[PNN]:https://arxiv.org/pdf/1611.00144.pdf
[WideDeep]:https://arxiv.org/pdf/1606.07792.pdf
[DeepFM]:https://arxiv.org/pdf/1703.04247.pdf
[PLM]:https://arxiv.org/abs/1704.05194
[DCN]:https://arxiv.org/abs/1708.05123
[AFM]:http://www.ijcai.org/proceedings/2017/435
[NFM]:https://arxiv.org/pdf/1708.05027.pdf
[xDeepFM]:https://arxiv.org/pdf/1803.05170.pdf
[AutoInt]:https://arxiv.org/abs/1810.11921
[ONN]:https://arxiv.org/pdf/1904.12579.pdf
[FGCNN]:https://arxiv.org/pdf/1904.04447
[FiBiNET]:https://arxiv.org/pdf/1905.09433.pdf
[FLEN]:https://arxiv.org/pdf/1911.04690.pdf
<!--
### Sequential Recommendation
| model/keywords | paper |
|:------|:------|
|GRU4Rec|Session-based Recommendations with Recurrent Neural Networks|
|Caser|Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding|
|DIN: Deep Interest Network|\[KDD 2018\][Deep Interest Network for Click-Through Rate Prediction][DIN]|
|Self-Attention |Next Item Recommendation with Self-Attention|
|Hierarchical Attention |Sequential Recommender System based on Hierarchical Attention Networks|
|DIEN: Deep Interest Evolution Network|\[AAAI 2019\][Deep Interest Evolution Network for Click-Through Rate Prediction][DIEN]|
|DISN: Deep Session Interest Network|\[IJCAI 2019\][Deep Session Interest Network for Click-Through Rate Prediction][DISN]|

[DIN]:https://arxiv.org/pdf/1706.06978.pdf
[DIEN]:https://arxiv.org/pdf/1809.03672.pdf
[DISN]:https://arxiv.org/abs/1905.06482

### Embedding Methods
| model | paper |
|:------|:------|
|node2vec|node2vec: Scalable Feature Learning for Networks|
|item2vec|ITEM2VEC: Neural item embedding for collaborative filtering|
|Airbnb embedding|Real-time Personalization using Embeddings for Search Ranking at Airbnb|
|EGES: Enhanced Graph Embedding with Side information|Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba|
-->