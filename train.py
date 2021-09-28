from abc import ABC

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, BertConfig, Trainer, AdamW, get_scheduler
from ofrecord_data_utils import OfRecordDataLoader
import oneflow as flow
import argparse
from tqdm import tqdm


class OneflowDataloaderToPytorchDataset(Dataset):
    def __init__(self, args):
        self.train_data_loader = OfRecordDataLoader(
            ofrecord_dir=args.ofrecord_path,
            mode="train",
            dataset_size=args.dataset_size,
            batch_size=args.train_batch_size,
            data_part_num=2,
            seq_length=args.seq_length,
            max_predictions_per_seq=args.max_predictions_per_seq,
        )
        self.dataset_size = args.dataset_size


    def __len__(self):
        return self.dataset_size


    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError
        of_data = self.train_data_loader()
        pt_data = dict()
        pt_data["input_ids"] = torch.tensor(of_data[0].numpy()).cuda()
        pt_data["token_type_ids"] = torch.tensor(of_data[3].numpy()).cuda()
        pt_data["attention_mask"] = torch.tensor(of_data[2].numpy()).cuda()
        label = torch.tensor(of_data[1].numpy()).cuda()
        masked_lm_ids = torch.tensor(of_data[4].numpy()).cuda()
        masked_lm_positions = torch.tensor(of_data[5].numpy()).cuda()
        masked_lm_weights = torch.tensor(of_data[6].numpy()).cuda()
        return pt_data, label, masked_lm_ids, masked_lm_positions, masked_lm_weights


class BertPreTrainingHeads(nn.Module):
    def __init__(self, hidden_size, vocab_size, hidden_act=nn.GELU()):
        super().__init__()
        self.predictions = BertLMPredictionHead(
            hidden_size, vocab_size, hidden_act)
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_scores = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_scores


class BertLMPredictionHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, hidden_act=nn.GELU()):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size, hidden_act)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.output_bias

    def forward(self, sequence_output):
        sequence_output = self.transform(sequence_output)
        sequence_output = self.decoder(sequence_output)
        return sequence_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size, hidden_act=nn.GELU()):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = hidden_act
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, sequence_output):
        sequence_output = self.dense(sequence_output)
        sequence_output = self.transform_act_fn(sequence_output)
        sequence_output = self.LayerNorm(sequence_output)
        return sequence_output


class PreTrainer(Trainer):
    def __init__(self, max_predictions_per_seq, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mlm_criterion = nn.CrossEntropyLoss(reduction="none")
        self.max_predictions_per_seq = max_predictions_per_seq

        def get_masked_lm_loss(
                logit_blob,
                masked_lm_positions,
                masked_lm_labels,
                label_weights,
                max_predictions_per_seq,
        ):
            # gather valid position indices
            logit_blob = torch.gather(
                logit_blob,
                index=masked_lm_positions.unsqueeze(2).to(
                    dtype=torch.int64).repeat(1, 1, 30522),
                dim=1,
            )
            logit_blob = torch.reshape(logit_blob, [-1, 30522])
            label_id_blob = torch.reshape(masked_lm_labels, [-1])

            # The `positions` tensor might be zero-padded (if the sequence is too
            # short to have the maximum number of predictions). The `label_weights`
            # tensor has a value of 1.0 for every real prediction and 0.0 for the
            # padding predictions.
            pre_example_loss = mlm_criterion(logit_blob, label_id_blob.long())
            pre_example_loss = torch.reshape(
                pre_example_loss, [-1, max_predictions_per_seq])
            sum_label_weight = torch.sum(label_weights, dim=-1)
            sum_label_weight = sum_label_weight / label_weights.shape[0]
            numerator = torch.sum(pre_example_loss * label_weights)
            denominator = torch.sum(label_weights) + 1e-5
            loss = numerator / denominator
            return loss

        self.ns_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.masked_lm_criterion = get_masked_lm_loss

    def compute_loss(self, model, cls, labels, id, pos, weight, inputs, return_outputs=False):
        outputs = model(**inputs)
        prediction_scores, seq_relationship_scores = cls(
            outputs.last_hidden_state, outputs.pooler_output)
        next_sentence_loss = self.ns_criterion(
            seq_relationship_scores.view(-1, 2), labels.long().view(-1)
        )

        masked_lm_loss = self.masked_lm_criterion(
            prediction_scores, pos, id, weight, max_predictions_per_seq=self.max_predictions_per_seq
        )

        total_loss = next_sentence_loss + masked_lm_loss
        return (total_loss, outputs) if return_outputs else total_loss


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
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    parser.add_argument(
        "--dataset_size", type=int, default=1024, help="The number of samples in an epoch cycle",
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
    parser.add_argument(
        "-s", "--seq_length", type=int, default=128, help="Maximum sequence len"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    parser.add_argument("--max_predictions_per_seq", type=int, default=20)
    parser.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate of adam")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )

    args = parser.parse_args()

    print("Creating Dataloader")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    configuration = BertConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               num_attention_heads=args.num_attention_heads, intermediate_size=4*args.hidden_size)
    model = BertModel(configuration).cuda()
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.adam_weight_decay)
    lr_scheduler = get_scheduler(
        "polynomial",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=300
    )
    cls = BertPreTrainingHeads(args.hidden_size, args.vocab_size).cuda()
    print("model structure", model)
    dataset = OneflowDataloaderToPytorchDataset(args)
    trainer = PreTrainer(
        max_predictions_per_seq=args.max_predictions_per_seq, model=model)
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    for i in range(args.epochs):
        for batch in tqdm(dataset):
            encoded_input.data, label, id, pos, weight = batch
            loss = trainer.compute_loss(
                model, cls, label, id, pos, weight, encoded_input)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
