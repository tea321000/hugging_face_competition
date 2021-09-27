# CCF BDCI BERT系统调优赛题baseline（Pytorch版本）

## 使用说明

运行train_BERT_base.sh和train_BERT_large.sh即可运行单机单卡的baseline。保持其它参数不变，通过调节shell文件里的hidden_size参数，即可观察不同hidden_size所占显存的变化（可通过`watch -n 0.1 nvidia-smi`直观观察）

```bash
python train.py \
--ofrecord_path sample_seq_len_512_example \
--lr 1e-4 --epochs 10 \
--train_batch_size 2 \
--seq_length=512 \
--max_predictions_per_seq=80 \
--num_hidden_layers=24 \
--num_attention_heads=16 \
--hidden_size=1024 \#要调节的参数
--vocab_size=30522
```
