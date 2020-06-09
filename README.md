# DeepRec-torch
Easy-to-use pytorch-based framework for RecSys models

## Model list
### Click Through Rate Prediction
| model | paper |
|:-----|:------|
|LR: Logistic Regression| [Simple and Scalable Response Prediction for Display Advertising][LR]|
|FM: Factorization Machine|\[ICDM 2010\][Factorization Machines][FM]|
|GBDT+LR: Gradient Boosting Tree with Logistic Regression|[Practical Lessons from Predicting Clicks on Ads at Facebook][GBDTLR]|
|CCPM: Convolutional Click Prediction Model|\[CIKM 2015\][A Convolutional Click Prediction Model][CCPM]|
|FFM: Field-aware Factorization Machine|\[RecSys 2016\][Field-aware Factorization Machines for CTR Prediction][FFM]|
|FNN: Factorization-supported Neural Network|\[ECIR 2016\][Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction][FNN]|
|PNN: Product-based Neural Network|\[ICDM 2016\][Product-based neural networks for user response prediction][PNN]|
|Wide and Deep|\[DLRS 2016\][Wide & Deep Learning for Recommender Systems][WideDeep]|
|DeepFM|\[IJCAI 2017\][DeepFM: A Factorization-Machine based Neural Network for CTR Prediction][DeepFM]|
|Piece-wise Linear Model|\[arxiv 2017\][Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction][PLM]|
|DCN: Deep & Cross Network|\[ADKDD 2017\][Deep & Cross Network for Ad Click Predictions][DCN]|
|AFM: Attentional Factorization Machine|\[IJCAI 2017\][Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks][AFM]|
|NFM: Neural Factorization Machine|\[SIGIR 2017\][Neural Factorization Machines for Sparse Predictive Analytics][NFM]|
|xDeepFM|\[KDD 2018\][xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems][xDeepFM]|
|AutoInt|\[arxiv 2018\][AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks][AutoInt]|
|ONN|\[arxiv 2019\][Operation-aware Neural Networks for User Response Prediction][ONN]|
|FGCNN|\[WWW 2019\][Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction][FGCNN]|
|FiBiNET|\[RecSys 2019\][FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction][FiBiNET]|
|FLEN|\[arxiv 2019\][FLEN: Leveraging Field for Scalable CTR Prediction][FLEN]|

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
