import torch
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset
from model.ctr.deepFM import DeepFM
from util.train import train_model_hold_out
from example.loader.criteo_loader import load_and_preprocess


def train_deepfm(x_idx, x_value, label, feat_meta, out_type='binary'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X_idx_tensor_gpu = torch.LongTensor(x_idx).to(device)
    X_value_tensor_gpu = torch.Tensor(x_value).to(device)
    y_tensor_gpu = torch.Tensor(label).to(device)
    y_tensor_gpu = y_tensor_gpu.reshape(-1, 1)

    dataset = TensorDataset(X_idx_tensor_gpu, X_value_tensor_gpu, y_tensor_gpu)
    deepfm_model = DeepFM(emb_dim=5, feat_dim=feat_meta.get_total_dim(), num_fields=feat_meta.get_num_fields(),
                          out_type=out_type).to(device)
    optimizer = torch.optim.Adam(deepfm_model.parameters(), lr=1e-4)

    train_model_hold_out(job_name='deepfm-binary-cls', device=device,
                         model=deepfm_model, dataset=dataset,
                         loss_func=nn.BCELoss(), optimizer=optimizer,
                         epochs=20, batch_size=256)
    return deepfm_model


if __name__ == '__main__':
    # load movielens dataset
    path = '../../dataset/criteo_sampled_10k.txt'
    X_idx, X_value, y, feature_meta = load_and_preprocess(path)
    model = train_deepfm(X_idx, X_value, y, feature_meta)
