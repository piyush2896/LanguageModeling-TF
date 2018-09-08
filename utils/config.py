config = {
    'nltk_packages': ['punkt', 'stopwords'],
    'tokenizer_path': 'tokenizers/punkt/english.pickle',
    'data_root': './dataset/',
    'vocab_size': 15000,
    'model_dir': './langmodel_got',
    'epochs': 20,
    'steps_per_epoch': 400,
    'batch_size': 16,
    'embedding_size': 128,
    'hidden_units': 128,
    'keep_rate': 0.7,
    'num_layers': 2,
    'lr': 1e-1,
    'ppdata_root': './ppdata',
    'max_gradient_norm': 10.
}
