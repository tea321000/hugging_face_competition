python train.py --ofrecord_path sample_seq_len_512_example --checkpoint_path . --lr 1e-4 --epochs 10 --train_batch_size 1 --val_batch_size 1 --seq_length=512 --max_predictions_per_seq=80 --num_hidden_layers=24 --num_attention_heads=16 --hidden_size=1024 --max_position_embeddings=512 --type_vocab_size=2 --vocab_size=30522 --attention_probs_dropout_prob=0.1 --hidden_dropout_prob=0.1 --hidden_size_per_head=64