import numpy as np
import argparse
from pathlib import Path
import umap
# from sklearn.decomposition import PCA
from ts2vec import TS2Vec
import datautils
from utils import init_dl_program


parser = argparse.ArgumentParser(description='Generate Representation for dora dataset and reduct dimension')
parser.add_argument('dataset',  type=str, help='The dataset name. default="dora"')
# parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
# parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
parser.add_argument('--dir_ck', type=str, default='training',  help='The root directory where the training progress of the model is stored')
parser.add_argument('--checkpoint', '-ck', type=str, default="model.pkl" ,  dest='ck', required=True, help='path of model checkpoint. end with ".pkl"')
parser.add_argument('--dir_embedding', type=str, default='embedding', help='The root directory where the training progress of the model is stored')

parser.add_argument('--gpu', type=int, default=1, help='The gpu no. used for training and inference (defaults to 1)')
parser.add_argument('--batch-size', type=int, default=1024, help='The batch size (defaults to 8)')
parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
parser.add_argument('--max-train-length', type=int, default=3000, 
                    help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
parser.add_argument('--input_dims', type=int, default=6, help='The Feature dimension (defaults to 6 for DORA sensor data. 1 mias univariable)')
parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')

parser.add_argument('--seed', type=int, default=None, help='The random seed')
parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
# parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
# parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')

parser.add_argument('--umap',  action="store_true", help='Whether to reduct dimension(UMAP)')

args = parser.parse_args()

print("Dataset:", args.dataset)
print("Arguments:", str(args))

config = dict(
    batch_size=args.batch_size,
    lr=args.lr,
    output_dims=args.repr_dims,
    max_train_length=args.max_train_length
)

device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

def main():
    
    # Load TS2Vec model 
    model = TS2Vec(
        input_dims=args.input_dims,
        device=device,
        **config
    )

    ck_ = Path(args.dir_ck)/args.ck
    print(f'Load Checpoint : {ck_}')
    model.load(ck_)

    # Load dataset
    train_data = datautils.load_DORA(args.dataset)
    print(f'Dataset Loaded : {train_data.shape}') 

    # Generate embedding
    # # Compute instance-level representations for dataset
    print(f'Generating embedding')
    embedding  = model.encode(train_data, encoding_window='full_series')  # n_instances x output_dims
    # embedding = embedding.cpu().numpy()
    print(f'Get embedding : {embedding.shape}') 
    
    #Save embedding
    file_  = ck_.parent.name + '_' + ck_.name
    file_ = file_.lstrip('dora_').rstrip('.pkl').replace('model_', 'ck')
    # print(f'{file_}')    
    save_path = Path(args.dir_embedding)
    save_path.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(save_path/f'embedding_{file_}.npz', embedding=embedding)
    print(f'{save_path}/embedding_{file_}.npz saved') 
    
    if args.umap ==True :
        print('='*50)
        print('\nStart UMAPing\n')
        n_sample = embedding.shape[0]
        n_neighbors = 200 if n_sample*0.1 > 200 else n_sample*0.1 
        n_neighbors
        
        for n in [2, 3]:
            reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n, random_state=0, verbose=True)
            umap_embed  = reducer.fit_transform(embedding)
            np.savez_compressed(save_path/f'umap{n}d{n_neighbors}n_embedding_{file_}.npz', embedding=umap_embed)
            print(f'{save_path}/umap{n}d{n_neighbors}n_embedding_{file_}.npz saved') 

        
if __name__ == '__main__':
    main()
    print("Finished.")