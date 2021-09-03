import csv
import argparse
import random
from pathlib import Path
from typing import DefaultDict, Union, Optional

import torch
import transformers as hf
import datasets as hfd
from tqdm import tqdm

from utils import Silence, HANS_subcases


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model-name', type=str, default='albert-xxlarge-v2')
parser.add_argument('-s', '--save-dir', type=Path, required=True)
parser.add_argument('--prompt-path', type=Path, required=True)
parser.add_argument('-e', '--epochs', type=int, default=5)
parser.add_argument('--few-shots', type=str, default='3,5,10,20,30,50,100')
parser.add_argument('--fully-train', action='store_true')
parser.add_argument('--train-batch-size', type=int, default=5)
parser.add_argument('--eval-batch-size', type=int, default=64)
parser.add_argument('--min-seen', type=int, default=300)
parser.add_argument('--eval-steps', type=int, default=20)
parser.add_argument('--seeds', type=str, default='42')
parser.add_argument('--experiment-name', type=str, default='')
parser.add_argument('--do-diagnosis', action='store_true')
parser.add_argument('--subsample-subcase', type=int, default=100)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--production', action='store_true', help='suppress training metrics printing')
args = parser.parse_args()
args.seeds = list(map(int, args.seeds.split(',')))
args.few_shots = list(map(int, args.few_shots.split(',')))
args.device = torch.device('cuda')
args.save_dir.mkdir(parents=True, exist_ok=True)

# import wandb
# wandb.login()
if args.production:
    hf.logging.set_verbosity_error()
    hfd.logging.set_verbosity_error()
else:
    import IPython
    from rich import inspect, traceback
    from rich.pretty import pprint
    traceback.install()


class Binary_NLI_Prompt():

    def __init__(self, row: dict):
        try:
            self.template = row['template']
            self.targets = row['targets']
        except KeyError:
            raise ValueError(f'Missing required prompt attributes: {row}')

        LM_targets = self.targets.rstrip().split(';')
        assert len(LM_targets) == 2
        self.label_to_word: dict[int, str] = {
            0: LM_targets[0],  # entailment
            1: LM_targets[1]   # non-entailment
        }

    # def __post_init__(self) -> None:
    #     self.word_to_label = {v: k for k, v in self.label_to_word.items()}


class PromptDataCollator(hf.DataCollatorForLanguageModeling):

    # from transformers.tokenization_utils_base import BatchEncoding

    def __init__(
            self,
            tokenizer: hf.PreTrainedTokenizerBase,
            label_to_word: dict[int, str],
            mlm: bool = True,
            mlm_probability: float = 0.15,
            pad_to_multiple_of: Optional[int] = None
            ) -> None:
        super().__init__(tokenizer, mlm, mlm_probability, pad_to_multiple_of)
        self.label_to_word = label_to_word

    def __call__(
            self,
            examples: list[dict[str, torch.Tensor]],
            ) -> dict[str, torch.Tensor]:
        targets = [
            self.tokenizer.encode(
                self.label_to_word[e['label']],
                add_special_tokens=False)[0]
            for e in examples]

        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, hf.tokenization_utils_base.BatchEncoding)):
            batch = self.tokenizer.pad(
                examples,
                return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            raise TypeError

        batch.pop('label')

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                inputs=batch["input_ids"],
                targets=torch.tensor(targets),
                special_tokens_mask=special_tokens_mask)
        else:
            raise ValueError
        return batch

    def mask_tokens(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            special_tokens_mask: Optional[torch.Tensor] = None
            ) -> tuple[torch.Tensor, torch.Tensor]:
        full_labels = inputs.clone()

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True) for val in full_labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        full_labels = torch.where(
            inputs == self.tokenizer.mask_token_id,
            torch.unsqueeze(targets, 1), -100)
        return inputs, full_labels


class PromptTrainer(hf.Trainer):

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
        if args.do_diagnosis:
            diagnosis = self.diagnostic_loop()
        else:
            diagnosis = {}
        diagnosis['step'] = self.state.global_step * args.train_batch_size  # TODO wrong when len(train) < batch size
        diagnosis['epoch'] = round(self.state.epoch, 1)

        # Validation Set
        correct = 0
        total = 0
        for batch in dev_dataloader:
            loss, logits, labels = self.prediction_step(self.model, batch, prediction_loss_only=False)
            pred_word_ids = logits.argmax(dim=-1)
            del loss, logits
            masked_indices = torch.where(batch['input_ids'] == self.tokenizer.mask_token_id, True, False)
            pred_word_ids = pred_word_ids.masked_select(masked_indices)
            labels = labels.masked_select(masked_indices)
            correct += (pred_word_ids == labels).sum().item()
            total += len(labels)
        # print(total, len(dev_dataloader))  # NOTE
        # assert total == len(dev_dataloader)
        dev_acc = round(correct / total, 4)
        diagnosis['dev_acc'] = dev_acc
        metrics = {'dev_acc': dev_acc}
        if args.debug:
            metrics['step'] = self.state.global_step * args.train_batch_size
            metrics['epoch'] = round(self.state.epoch, 1)
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
            dataloader = self.get_eval_dataloader(subset)
            total = 0
            correct = 0
            for batch in dataloader:
                _, logits, labels = self.prediction_step(self.model, batch, prediction_loss_only=False)
                pred_word_ids = logits.argmax(dim=-1)
                del logits
                masked_indices = torch.where(batch['input_ids'] == self.tokenizer.mask_token_id, True, False)
                pred_word_ids = pred_word_ids.masked_select(masked_indices)
                labels = labels.masked_select(masked_indices)
                correct += (pred_word_ids == labels).sum().item()
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
        prompt: Binary_NLI_Prompt,
        train_set: hfd.Dataset,
        dev_set: hfd.Dataset,
        diagnostic_set: hfd.Dataset,
        ) -> list[dict]:
    if len(train_set) <= args.train_batch_size:
        # args._train_batch_size = len(train_set)  # a batch has the entire train set
        eval_strategy = 'steps'
        batched_eval_steps = args.eval_steps // len(train_set)
        epochs = max(1, args.min_seen // len(train_set))
    elif len(train_set) <= 256:  # < args.min_seen:  # few_shots
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
        disable_tqdm=args.production,
        report_to=None,
        # report_to='wandb',
        # run_name=f'{prompt.template}_{prompt.targets}_s{args.current_seed}',
        save_strategy='no',
        seed=args.current_seed
    )

    tqdm.write(f'\n{prompt.template} {prompt.targets}')
    tqdm.write(f"{train_set[0]['idx']} {train_set[0]['hypothesis']}")
    tqdm.write(f"{train_set[-1]['idx']} {train_set[-1]['hypothesis']}")

    prompt_collator = PromptDataCollator(
        tokenizer=tokenizer, label_to_word=prompt.label_to_word)
    trainer = PromptTrainer(
        model=model,
        args=train_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        data_collator=prompt_collator,
        tokenizer=tokenizer)
    trainer.diagnostic_set = diagnostic_set  # hack
    trainer.diagnoses = []
    with Silence(suppress_stdout=args.production, suppress_stderr=args.production):
        trainer.train()
    # wandb.finish()
    return trainer.diagnoses


def check_conflicting_targets(
        tokenizer: hf.AutoTokenizer,
        label_to_word: dict[int, str]
        ) -> bool:
    possible_duplicates = set()
    for word in label_to_word.values():
        token = tokenizer.encode(word, add_special_tokens=False)
        token = tokenizer.convert_ids_to_tokens(token)
        if len(token) > 1:
            raise ValueError(f'Target word {word} is multi-token: {token}')
        token = token[0]
        if token == tokenizer.unk_token:
            raise ValueError(f'Target word {word} is the UNK token')
        if token in possible_duplicates:
            raise ValueError(f'Target word {word} as {token} has duplicates')
        possible_duplicates.add(token)
    return False


def main() -> None:
    tokenizer = hf.AutoTokenizer.from_pretrained(args.model_name)
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
    out_path = args.save_dir / f'{args.experiment_name}_s{args.current_seed}.csv'
    out_file = open(out_path, 'w')
    writer = csv.DictWriter(out_file, fieldnames=manually_ordered_fieldnames)
    writer.writeheader()

    prompts: list[Binary_NLI_Prompt] = []
    with open(args.prompt_path) as p_file:
        reader = csv.DictReader(p_file)
        for row in reader:
            if args.experiment_name and row['experiment'] != args.experiment_name:
                continue
            prompt = Binary_NLI_Prompt(row)
            try:
                check_conflicting_targets(tokenizer, prompt.label_to_word)
                assert '{mask}' in prompt.template
                assert '{premise}' in prompt.template
                assert '{hypothesis}' in prompt.template
            except (ValueError, AssertionError) as err_msg:
                manually_ordered_fieldnames.append('error')
                writer.writerow({
                    'brand': args.model_name,
                    'template': prompt.template,
                    'targets': prompt.targets,
                    'error': str(err_msg)})
                out_file.flush()
                continue
            prompts.append(prompt)
            print(prompt.template, prompt.targets, flush=True)
        if args.debug:  # only train on a small subset
            prompts = prompts[:1]

    for prompt in tqdm(prompts, desc='Prompts'):
        def fill_template_and_tokenize(data: dict) -> dict:
            filled_template = prompt.template.format(
                premise=data['premise'],
                hypothesis=data['hypothesis'],
                mask=tokenizer.mask_token)
            return tokenizer(
                filled_template, truncation=True, padding=True,
                return_special_tokens_mask=False)

        proc_train = train_set.map(fill_template_and_tokenize)
        proc_dev = dev_set.map(fill_template_and_tokenize)
        if args.do_diagnosis:
            proc_diag = diagnostic_set.map(fill_template_and_tokenize)
        else:
            proc_diag = None
        # proc_train = proc_train.remove_columns(['idx', 'premise', 'hypothesis'])
        proc_dev = proc_dev.remove_columns(['idx', 'premise', 'hypothesis'])

        for num_shots in tqdm(args.few_shots, desc='Num. Shots', disable=len(args.few_shots) == 1):
            result_table: list[dict] = []
            max_index = len(train_set) - num_shots
            start_index = random.randint(0, max_index)
            sample_indices = range(start_index, start_index + num_shots)
            k_shot_proc_train = proc_train.select(sample_indices)

            model = hf.AutoModelForMaskedLM.from_pretrained(args.model_name)
            setup_info: dict[str, Union[str, int, float]] = {
                'brand': args.model_name,
                # 'm. param.': f'{model.num_parameters() / 1_000_000:.0f}',
                'template': prompt.template,
                'targets': prompt.targets,
                'train_size': num_shots,
                'starting_example_index': start_index,
                'seed': args.current_seed,
            }
            diagnoses = arrange_training(
                model, tokenizer, prompt,
                k_shot_proc_train, proc_dev, proc_diag)
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
    # End looping prompts
    out_file.close()


if __name__ == '__main__':
    for seed in tqdm(args.seeds, desc='Random Seeds', disable=len(args.seeds) == 1):
        hf.set_seed(seed)
        args.current_seed = seed
        main()
