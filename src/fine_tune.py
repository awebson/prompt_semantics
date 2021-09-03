import csv
import argparse
import random
from pathlib import Path
from typing import DefaultDict, Union, Optional

import torch
import transformers as hf
import datasets as hfd
# import numpy as np
from tqdm import tqdm

from utils import Silence, HANS_subcases


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save-dir', type=Path, required=True)
parser.add_argument('-e', '--epochs', type=int, default=5)
parser.add_argument('--few-shots', type=str, default='3,5,10,20,30,50,100,250')
parser.add_argument('--fully-train', action='store_true')
parser.add_argument('--train-batch-size', type=int, default=5)
parser.add_argument('--eval-batch-size', type=int, default=64)
parser.add_argument('--min-seen', type=int, default=300)
parser.add_argument('--eval-steps', type=int, default=20)
parser.add_argument('--seeds', type=str, default='42')
# parser.add_argument('--do-diagnosis', action='store_true')
parser.add_argument('--subsample-subcase', type=int, default=100)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--production', action='store_true')
args = parser.parse_args()
args.seeds = list(map(int, args.seeds.split(',')))
args.few_shots = list(map(int, args.few_shots.split(',')))
args.device = torch.device('cuda')
args.save_dir.mkdir(parents=True, exist_ok=True)
args.model_brands = [
    # 'distilbert-base-uncased',
    # 'bert-large-uncased',
    # 'roberta-large',
    'albert-xxlarge-v2',
]

hfd.logging.set_verbosity_error()
if not args.production:
    import IPython
    from rich import traceback, inspect
    traceback.install()
else:
    hf.logging.set_verbosity_error()

# if args.debug:  # only train on a small subset
#     args.few_shots = [32,]
#     args.model_brands = args.model_brands[:1]


class DiagnosticTrainer(hf.Trainer):

    def evaluation_loop(
            self,
            dev_dataloader: torch.utils.data.DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[list[str]] = None,
            metric_key_prefix: str = "eval",
            ) -> hf.trainer_utils.EvalLoopOutput:
        # model = self._wrap_model(self.model, training=False)
        self.model.eval()
        diagnosis = self.diagnostic_loop()
        diagnosis['step'] = self.state.global_step * args.train_batch_size  # TODO wrong when len(train) < batch size
        diagnosis['epoch'] = round(self.state.epoch, 1)

        # Validation Set
        correct = 0
        total = 0
        for batch in dev_dataloader:
            loss, logits, labels = self.prediction_step(self.model, batch, prediction_loss_only=False)
            predictions = logits.argmax(dim=-1)
            del loss, logits
            correct += (predictions == labels).sum().item()
            total += len(labels)
        dev_acc = round(correct / total, 4)
        metrics = {'dev_acc': dev_acc}
        diagnosis['dev_acc'] = dev_acc
        self.diagnoses.append(diagnosis)
        return hf.trainer_utils.EvalLoopOutput(
            predictions=None, label_ids=None, metrics=metrics, num_samples=len(dev_dataloader))

    def diagnostic_loop(self) -> dict:
        row = {}
        naive_average = 0.0
        per_case_average: DefaultDict[str, float] = DefaultDict(float)
        for subcase in HANS_subcases:
            case = subcase.split('_')[0].upper()
            subset = self.diagnostic_set.filter(lambda e: e['subcase'] == subcase)
            if args.subsample_subcase:
                subset = subset.shuffle().select(range(args.subsample_subcase))
            # subset = subset.remove_columns(['premise', 'hypothesis', 'heuristic', 'subcase', 'template'])
            dataloader = self.get_eval_dataloader(subset)
            total = 0
            correct = 0
            for batch in dataloader:
                loss, logits, labels = self.prediction_step(self.model, batch, prediction_loss_only=False)
                predictions = logits.argmax(dim=-1)
                del loss, logits
                correct += (predictions == labels).sum().item()
                total += len(labels)
            # End looping batches

            subcase_accuracy = correct / total
            row[subcase] = round(subcase_accuracy, 4)
            per_case_average[case] += subcase_accuracy
            naive_average += subcase_accuracy
        # End looping subcases
        row |= {case: round(acc / 5, 4) for case, acc in per_case_average.items()}
        row['diag_avg'] = round(naive_average / len(HANS_subcases), 4)
        return row


def arrange_training(
        model: hf.AutoModelForMaskedLM,
        tokenizer: hf.AutoTokenizer,
        train_set: hfd.Dataset,
        dev_set: hfd.Dataset,
        diagnostic_set: hfd.Dataset,
        ) -> list[dict]:
    if len(train_set) <= args.train_batch_size:
        eval_strategy = 'steps'
        batched_eval_steps = args.eval_steps // len(train_set)
        epochs = max(1, args.min_seen // len(train_set))
    elif len(train_set) <= 100:  # < args.min_seen:  # few_shots
        eval_strategy = 'steps'  # batched steps, actually
        batched_eval_steps = args.eval_steps // args.train_batch_size
        epochs = max(1, args.min_seen // len(train_set))
    else:
        eval_strategy = 'epoch'
        epochs = args.epochs
        batched_eval_steps = 0

    train_args = hf.TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=False,
        evaluation_strategy=eval_strategy,
        eval_steps=batched_eval_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=epochs,
        learning_rate=1e-5,
        weight_decay=0.01,
        remove_unused_columns=True,
        disable_tqdm=not args.debug,
        save_strategy='no',
        seed=args.current_seed
    )

    # metric = hfd.load_metric('super_glue', 'rte')
    # def compute_metrics(eval_pred):
    #     logits, labels = eval_pred
    #     predictions = np.argmax(logits, axis=-1)
    #     return metric.compute(predictions=predictions, references=labels)

    trainer = DiagnosticTrainer(
        model=model,
        args=train_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics
    )
    trainer.diagnostic_set = diagnostic_set  # hack
    trainer.diagnoses = []
    with Silence(suppress_stdout=args.production, suppress_stderr=args.production):
        trainer.train()
    return trainer.diagnoses


def main() -> None:
    train_set = hfd.load_dataset('super_glue', 'rte', split='train')
    dev_set = hfd.load_dataset('super_glue', 'rte', split='validation')
    diagnostic_set = hfd.load_dataset('hans', split='validation')
    diagnostic_set = diagnostic_set.remove_columns(
        ['parse_premise', 'parse_hypothesis', 'binary_parse_premise', 'binary_parse_hypothesis'])
    if args.fully_train:
        args.few_shots.append(len(train_set))

    manually_ordered_fieldnames = [
        'template',
        'targets',
        'train_size',
        'epoch',
        'step',
        'dev_acc',
        'diag_avg',
        'LE', 'SE', 'CE',
        'LN', 'SN', 'CN',
        'starting_example_index',
        'seed',
        'brand',
    ]
    manually_ordered_fieldnames += HANS_subcases
    out_path = args.save_dir / f'fine_tuned_s{args.current_seed}.tsv'
    out_file = open(out_path, 'w')
    writer = csv.DictWriter(
        out_file, fieldnames=manually_ordered_fieldnames, dialect=csv.excel_tab)
    writer.writeheader()

    for brand in tqdm(args.model_brands, desc='Model Brands'):
        tokenizer = hf.AutoTokenizer.from_pretrained(brand)
        def tokenize_sentence_pair(examples: dict) -> dict:
            return tokenizer(
                examples['premise'], examples['hypothesis'], truncation=True,
                padding=True, return_special_tokens_mask=True)

        proc_train = train_set.map(tokenize_sentence_pair)
        proc_dev = dev_set.map(tokenize_sentence_pair)
        proc_diag = diagnostic_set.map(tokenize_sentence_pair)

        for num_shots in tqdm(args.few_shots, desc='Num. Shots'):
            result_table: list[dict] = []
            max_index = len(train_set) - num_shots
            start_index = random.randint(0, max_index)
            sample_indices = range(start_index, start_index + num_shots)
            k_shot_proc_train = proc_train.select(sample_indices)

            model = hf.AutoModelForSequenceClassification.from_pretrained(brand)
            setup_info: dict[str, Union[str, int, float]] = {
                'brand': brand,
                # 'm. param.': f'{model.num_parameters() / 1_000_000:.0f}',
                'template': 'FT',
                'targets': 'FT;FT',
                'train_size': num_shots,
                'starting_example_index': start_index,
                'seed': args.current_seed,
            }
            diagnoses = arrange_training(
                model, tokenizer, k_shot_proc_train, proc_dev, proc_diag)
            result_table += [setup_info | diagnosis for diagnosis in diagnoses]
            del model
            torch.cuda.empty_cache()

            for row in result_table:
                for column in row.keys():
                    if column not in manually_ordered_fieldnames:
                        manually_ordered_fieldnames.append(column)
            writer.writerows(result_table)
            out_file.flush()
        # End looping num_shots
    # End looping model brands
    out_file.close()


if __name__ == '__main__':
    for seed in tqdm(args.seeds, desc='Random Seeds', disable=len(args.seeds) == 1):
        hf.set_seed(seed)
        args.current_seed = seed
        main()
