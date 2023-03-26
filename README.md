
 

<img src=workflow.png width=600 />

[Review for  Deep Learning with Time Series Data](https://hackmd.io/@YungHuiHsu/r1yCihCPo)

[TS2Vec(Towards Universal Representation of Time Series) 論文筆記](https://hackmd.io/SL6FT2JISlykJFhDognKyg)

1. Datasets for training  
    - The data source uses Chung-Hao Lee's pre-screened anomalous events([ADDCAT](https://arxiv.org/abs/2212.07691))
        - `ts2vec/datasets/DORA/df_event_all_v1.feather`
    - data preparation  
        - `transform_data_CHPrep.ipynb`
        - output at : `ts2vec/datasets/DORA/`
            - data : `dora.npz` (n, length, dim)
            - meta : `dora_meta.csv`
    - Pre-processing process for Finetune 
        - `ts2vec/datasets/prepare_DORA_KA51AG8742.ipynb`
            - `X_train.npz, X_test.npz, y_train.npz, y_test.npz`

2. Training (in directory ts2vec/)  
We use the self-supervised learning model ts2vec for training and feature extraction.
    - Pretrain commands
        ```bash
        python3 train.py dora  training  --loader DORA --batch-size  16 --repr-dims 320 --epochs 500
        ```
    - Get the embedding and dimension reduction 
        ```bash
        python3 get_embedding.py  dora  --checkpoint  dora_b16_320d/model_500.pkl  --repr-dims 320  --batch-size 1024  --umap
        ```

3. Analysis and auto-labeling
- Representation Analysis
    - `ts2vec/explore_feature.ipynb`
    - Validate appropriate hyperparameter settings
    - Explore and analyze the representation  
- Finetune and get more label
    - `ts2vec/verify_label.ipynb` 
    - Fine-tuning with vehicle_model : "KA51AG8742" datasets (with labels)
        - train the classification model to detect anomalies
    - Use fine-tuning model to infer all data and obtain the obtain labels for abnormal events (automatic labeling)
