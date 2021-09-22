from abc import ABC

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, BertConfig, Trainer
from ofrecord_data_utils import OfRecordDataLoader
import oneflow as flow
import argparse
from tqdm import tqdm


class OneflowDataloaderToPytorchDataset(Dataset):
    def __init__(self, args):
        self.train_data_loader = OfRecordDataLoader(
            ofrecord_dir=args.ofrecord_path,
            mode="train",
            dataset_size=1024,
            batch_size=args.train_batch_size,
            data_part_num=1,
            seq_length=args.seq_length,
            max_predictions_per_seq=args.max_predictions_per_seq,
        )

    def __len__(self):
        return 1024

    def __getitem__(self, _):
        of_data = self.train_data_loader()
        pt_data = dict()
        pt_data["input_ids"] = torch.tensor(of_data[0].numpy()).cuda()
        pt_data["token_type_ids"] = torch.tensor(of_data[3].numpy()).cuda()
        pt_data["attention_mask"] = torch.tensor(of_data[2].numpy()).cuda()
        label = torch.tensor(of_data[1].numpy()).cuda()
        return pt_data, label


class MultilabelTrainer(Trainer):
    def compute_loss(self, model, labels, inputs, return_outputs=False):
        # labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.pooler_output.argmax(-1)
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, 1).float(),
                        labels.float())
        # print(loss)
        return (loss, outputs) if return_outputs else loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ofrecord_path",
        type=str,
        default="wiki_ofrecord_seq_len_128_example",
        help="Path to ofrecord dataset",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=32, help="Validation batch size"
    )

    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    parser.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    # parser.add_argument(
    #     "--intermediate_size",
    #     type=int,
    #     default=3072,
    #     help="intermediate size of bert encoder",
    # )
    parser.add_argument("--max_position_embeddings", type=int, default=512)
    parser.add_argument(
        "-s", "--seq_length", type=int, default=128, help="Maximum sequence len"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    parser.add_argument("--type_vocab_size", type=int, default=2)
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--hidden_size_per_head", type=int, default=64)
    parser.add_argument("--max_predictions_per_seq", type=int, default=20)
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")

    parser.add_argument(
        "--with-cuda",
        type=bool,
        default=True,
        help="Training with CUDA: true, or false",
    )
    parser.add_argument(
        "--cuda_devices", type=int, nargs="+", default=None, help="CUDA device ids"
    )

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of adam")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="Adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="Adam first beta value"
    )
    parser.add_argument(
        "--print_interval", type=int, default=10, help="Interval of printing"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints",
        help="Path to model saving",
    )

    args = parser.parse_args()

    if args.with_cuda:
        device = flow.device("cuda")
    else:
        device = flow.device("cpu")

    print("Device is: ", device)

    print("Creating Dataloader")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    configuration = BertConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, num_attention_heads = args.num_attention_heads)
    model = BertModel(configuration)
    print("pytorch model", model)
    dataset = OneflowDataloaderToPytorchDataset(args)
    trainer = MultilabelTrainer(model=model)
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    # output = model(**encoded_input)
    for i in range(args.epochs):
        for batch in tqdm(dataset):
            encoded_input.data, label = batch
            trainer.compute_loss(model, label, encoded_input)

    # train_data_loader = OfRecordDataLoader(
    #     ofrecord_dir=args.ofrecord_path,
    #     mode="train",
    #     dataset_size=1024,
    #     batch_size=args.train_batch_size,
    #     data_part_num=1,
    #     seq_length=args.seq_length,
    #     max_predictions_per_seq=args.max_predictions_per_seq,
    # )
    # model = BertModel.from_pretrained("bert-base-uncased")
    # for epoch in range(args.epochs):
    #     for batch in train_data_loader:
    #         print(batch)


