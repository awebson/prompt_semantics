import csv
import argparse
import random
import gc
from pathlib import Path
from typing import DefaultDict, Union, Optional, Any

import torch
import transformers as hf
import datasets as hfd
from tqdm import tqdm

from utils import table_columns


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model-name', type=str, default='albert-xxlarge-v2')
parser.add_argument('-s', '--save-dir', type=Path, required=True)
parser.add_argument(
    '-d', '--dataset', type=str, required=True,
    choices=['rte', 'cb', 'anli', 'mnli'])
parser.add_argument('-pp', '--prompt-path', type=Path, required=True)
parser.add_argument('-ep', '--epochs', type=int, default=16)
parser.add_argument('-ns', '--num-shots', type=str, default='1,2,4,8,16,32,64,128,256')
parser.add_argument('--fully-train', action='store_true')
parser.add_argument('--non-rank-eval', action='store_true')
parser.add_argument('-tb', '--train-batch-size', type=int, default=8)
parser.add_argument('-eb', '--eval-batch-size', type=int, default=16)  # 64 max
parser.add_argument('-ga', '--grad-accumulation', type=int, default=1)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-5)
parser.add_argument('--fixed-num-evals', type=int, default=0)
parser.add_argument('--seeds', type=str, default='42')
parser.add_argument('-exp', '--experiment-name', type=str, required=True)
parser.add_argument('--do-diagnosis', action='store_true')
parser.add_argument('--subsample-subcase', type=int, default=100)  # only for HANS diagnosis
parser.add_argument('--debug', action='store_true')
parser.add_argument('-wb', '--wandb-proj', type=str, default=None)
parser.add_argument('-wbn', '--wandb-name', type=str, default='')
parser.add_argument('--production', action='store_true', help='suppress training metrics printing')
args = parser.parse_args()
args.seeds = list(map(int, args.seeds.split(',')))
args.num_shots = list(map(int, args.num_shots.split(',')))
args.save_dir.mkdir(parents=True, exist_ok=True)

if args.wandb_proj:
    import wandb

if args.production:
    hf.logging.set_verbosity(50)
    hfd.logging.set_verbosity(50)
    # wandb.login()
else:
    import IPython
    from rich import inspect, traceback
    from rich.pretty import pprint
    traceback.install()

# hf.logging.set_verbosity(10)
# hfd.logging.set_verbosity(10)
# hf.logging.enable_explicit_format()

class NLIPrompt:

    def __init__(self, row: dict, dataset: str):
        """dataset should be one among ('rte', 'cb', 'anli', 'mnli')"""
        try:
            self.template = row['template']
            self.targets = row['targets']
            self.template_name = row['template_name']
            self.template_category = row['template_category']
            self.target_category = row['target_category']
            self.comment = row['comment']
        except KeyError:
            raise ValueError(f'Missing required prompt attributes: {row}')

        if '\\n' in self.template:
            # print('Detected escaped newline in a template.')
            self.template = self.template.replace('\\n', '\n')

        if '{mask}' not in self.template:
            self.template += ' {mask}'

        LM_targets = self.targets.rstrip().split(';')
        self.class_id_to_word: dict[int, str]
        if dataset in ('anli', 'mnli', 'cb'):
            self.ternary = True
            try:
                assert len(LM_targets) == 3
            except:
                pprint(row)
            if dataset != 'cb':
                self.class_id_to_word = {
                    0: LM_targets[0],  # entailment
                    1: LM_targets[1],  # neutral
                    2: LM_targets[2]}  # contradiction
            else:
                self.class_id_to_word = {
                    0: LM_targets[0],  # entailment
                    1: LM_targets[2],  # contradiction
                    2: LM_targets[1]}  # neutral
        elif dataset in ('rte', 'hans'):
            self.ternary = False
            assert len(LM_targets) == 2
            self.class_id_to_word = {
                0: LM_targets[0],  # entailment
                1: LM_targets[1]}  # non-entailment
        else:
            raise ValueError('Unknown NLI dataset.')

    def check_conflicting_targets(self, tokenizer: hf.AutoTokenizer) -> bool:
        """assume the tokenizer is unchanged throughout this script"""
        self.class_id_to_word_id: dict[int, int] = {}
        possible_duplicates = set()
        for class_id, word in self.class_id_to_word.items():
            token_id = tokenizer.encode(word, add_special_tokens=False)
            self.class_id_to_word_id[class_id] = token_id[0]
            token = tokenizer.convert_ids_to_tokens(token_id)
            if len(token) > 1:
                raise ValueError(f'Target word {word} is multi-token: {token}')
            token = token[0]
            if token == tokenizer.unk_token:
                raise ValueError(f'Target word {word} is the UNK token')
            if token in possible_duplicates:
                raise ValueError(f'Target word {word} as {token} has duplicates')
            possible_duplicates.add(token)
        return False

    def __str__(self):
        return self.template + ' -> ' + str(self.class_id_to_word)


class PromptDataCollator(hf.DataCollatorForLanguageModeling):

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        target_word_ids = torch.tensor(
            [self.class_id_to_word_id[e['label']] for e in examples])
        batch = self.tokenizer.pad(
            examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        batch['int_class_labels'] = batch.pop('label')
        batch['labels'] = torch.where(
            batch['input_ids'] == self.tokenizer.mask_token_id,
            torch.unsqueeze(target_word_ids, 1), -100)
        return batch


class PromptTrainer(hf.Trainer):

    def training_step(
            self, model, inputs
            ) -> torch.Tensor:
        if 'int_class_labels' in inputs:
            del inputs['int_class_labels']
        return super().training_step(model, inputs)

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
        diagnosis['epoch'] = self.state.epoch
        diagnosis['step'] = self.state.global_step
        # diagnosis['cumulative_examples'] = self.state.global_step * self.adjusted_train_batch_size

        # Validation Set
        correct = 0
        total = 0
        for batch in tqdm(dev_dataloader, desc='Evaluating Dev', disable=not args.debug):
            int_class_labels = batch.pop('int_class_labels')
            loss, logits, labels = self.prediction_step(self.model, batch, prediction_loss_only=False)
            device = logits.device  # HACK
            masked_indices = torch.eq(batch['input_ids'], self.tokenizer.mask_token_id).to(device)
            if args.non_rank_eval:
                pred_word_ids = logits.argmax(dim=-1)  # batch * seq * vocab
                # del loss, logits
                pred_word_ids = pred_word_ids.masked_select(masked_indices)
                labels = labels.masked_select(masked_indices)
                correct += (pred_word_ids == labels).sum().item()
            else:
                batch_size = logits.shape[0]
                logits = logits.masked_select(masked_indices.unsqueeze(-1))  # only need logits of the [mask] token
                logits = logits.view(batch_size, -1)  # batch_size * max_seq_len
                class_logits = logits[:,self.label_word_ids]  # (ent_id, neu_id, cont_id)
                pred_class = class_logits.argmax(dim=-1)  # choose class with largest logits
                if self.args._n_gpu > 0:
                    int_class_labels = int_class_labels.to('cuda')  # NOTE
                correct += (pred_class == int_class_labels).sum().item()
            total += len(labels)
        dev_acc = round(correct / total, 4)
        diagnosis['eval_rank_acc'] = dev_acc
        metrics = {'eval_rank_acc': dev_acc}
        if args.debug:
            metrics['epoch'] = self.state.epoch
            metrics['step'] = self.state.global_step
            # metrics['cumulative_examples'] = self.state.global_step * self.adjusted_train_batch_size
            metrics['num. eval examples'] = total
        self.diagnoses.append(diagnosis)
        return hf.trainer_utils.EvalLoopOutput(
            predictions=None, label_ids=None, metrics=metrics, num_samples=len(dev_dataloader))

    # def diagnostic_loop(self) -> dict:
    #     row = {}
    #     naive_average = 0.0
    #     per_case_average: DefaultDict[str, float] = DefaultDict(float)
    #     for subcase in HANS_subcases:
    #         case = subcase.split('_')[0].upper()
    #         subset = self.diagnostic_set.filter(lambda e: e['subcase'] == subcase)
    #         if args.subsample_subcase:
    #             subset = subset.shuffle().select(range(args.subsample_subcase))
    #         dataloader = self.get_eval_dataloader(subset)
    #         total = 0
    #         correct = 0
    #         for batch in dataloader:
    #             _, logits, labels = self.prediction_step(self.model, batch, prediction_loss_only=False)
    #             pred_word_ids = logits.argmax(dim=-1)
    #             del logits
    #             masked_indices = torch.where(batch['input_ids'] == self.tokenizer.mask_token_id, True, False)
    #             pred_word_ids = pred_word_ids.masked_select(masked_indices)
    #             labels = labels.masked_select(masked_indices)
    #             correct += (pred_word_ids == labels).sum().item()
    #             total += len(labels)
    #         # End looping batches

    #         subcase_accuracy = correct / total
    #         row[subcase] = round(subcase_accuracy, 4)
    #         per_case_average[case] += subcase_accuracy
    #         naive_average += subcase_accuracy
    #     # End looping subcases
    #     row |= {case: round(acc / 5, 4) for case, acc in per_case_average.items()}
    #     row['diag_avg'] = round(naive_average / len(HANS_subcases), 4)
    #     return row


def arrange_training(
        model: hf.AutoModelForMaskedLM,
        tokenizer: hf.AutoTokenizer,
        prompt: NLIPrompt,
        train_set: hfd.Dataset,
        dev_set: hfd.Dataset,
        diagnostic_set: hfd.Dataset,
        ) -> list[dict]:
    if len(train_set) <= args.train_batch_size:
        adjusted_train_batch_size = len(train_set)  # a batch has the entire train set
        eval_strategy = 'epoch'
        epochs = args.epochs * 2
        batched_eval_steps = None
    else:
        adjusted_train_batch_size = args.train_batch_size
        eval_strategy = 'epoch'
        epochs = args.epochs
        batched_eval_steps = None

    if args.fixed_num_evals:
        eval_strategy = 'steps'
        total_steps = len(train_set) * epochs / adjusted_train_batch_size
        batched_eval_steps = total_steps / args.fixed_num_evals

    num_shot = len(train_set) if train_set is not None else 0
    save_dir = args.save_dir / prompt.template_name
    save_dir.mkdir(parents=True, exist_ok=True)
    train_args = hf.TrainingArguments(
        output_dir=save_dir,
        overwrite_output_dir=False,
        evaluation_strategy=eval_strategy,
        eval_steps=batched_eval_steps,
        per_device_train_batch_size=adjusted_train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        num_train_epochs=epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=8 if args.production else 1,
        remove_unused_columns=True,
        disable_tqdm=args.production,
        log_level='info' if args.debug else 'warning',
        report_to='wandb' if args.wandb_proj else None,
        run_name=f'{args.wandb_name}S{num_shot} {prompt.template_name} R{args.current_seed}',
        save_strategy='epoch',
        seed=args.current_seed
    )

    tqdm.write(f'\n{prompt}')
    # tqdm.write(f"{train_set[0]['idx']} {train_set[0]['hypothesis']}")
    # tqdm.write(f"{train_set[-1]['idx']} {train_set[-1]['hypothesis']}")

    prompt_collator = PromptDataCollator(tokenizer=tokenizer)
    prompt_collator.class_id_to_word_id = prompt.class_id_to_word_id

    if args.wandb_proj:
        reminder = {
            '_template_name': prompt.template_name,
            '_template': prompt.template,
            '_targets': prompt.targets,
            '_comment': prompt.comment,
            '_num_shot': num_shot,
        }
        wandb.init(
            name=f'{args.wandb_name}S{num_shot} {prompt.template_name} R{args.current_seed}',
            project=args.wandb_proj,
            config=reminder,
            reinit=True)

    trainer = PromptTrainer(
        model=model,
        args=train_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        data_collator=prompt_collator,
        tokenizer=tokenizer)

    # minor hacks
    trainer.adjusted_train_batch_size = adjusted_train_batch_size
    trainer.label_word_ids = [wid for wid in prompt.class_id_to_word_id.values()]
    trainer.diagnostic_set = diagnostic_set
    trainer.diagnoses = []

    trainer.train()
    if args.wandb_proj:
        wandb.finish()
    return trainer.diagnoses


def main() -> None:
    if args.dataset in ('rte', 'cb'):
        train_set = hfd.load_dataset('super_glue', args.dataset, split='train')
        dev_set = hfd.load_dataset('super_glue', args.dataset, split='validation')
    elif args.dataset == 'anli':
        train_set = hfd.load_dataset('anli', split='train_r1')
        dev_set = hfd.load_dataset('anli', split='dev_r1')
    elif args.dataset == 'mnli':
        train_set = hfd.load_dataset('multi_nli', split='train')
        dev_set = hfd.load_dataset('multi_nli', split='validation_matched')
        # dev_set = hfd.load_dataset('multi_nli', split='validation_mismatched')
    else:
        raise ValueError('Unknown NLI dataset.')

    if args.fully_train:
        args.num_shots.append(len(train_set))
    if args.do_diagnosis:
        diagnostic_set = hfd.load_dataset('hans', split='validation')
        diagnostic_set = diagnostic_set.remove_columns(
            ['parse_premise', 'parse_hypothesis', 'binary_parse_premise', 'binary_parse_hypothesis'])
        # manually_ordered_fieldnames += ['diag_avg', 'LE', 'SE', 'CE', 'LN', 'SN', 'CN']
        # manually_ordered_fieldnames += HANS_subcases

    out_path = args.save_dir / f'{args.experiment_name}_s{args.current_seed}.csv'
    out_file = open(out_path, 'w')
    writer = csv.DictWriter(out_file, fieldnames=table_columns)
    writer.writeheader()

    tokenizer = hf.AutoTokenizer.from_pretrained(args.model_name)
    prompts: list[NLIPrompt] = []
    with open(args.prompt_path) as p_file:
        reader = csv.DictReader(p_file)
        for row in reader:
            if row['template'] == '':
                continue
            prompt_exp_names = row['experiment'].split(';')
            arg_exp_names = args.experiment_name.split(';')
            if all([arg_exp not in prompt_exp_names for arg_exp in arg_exp_names]):
                continue
            prompt = NLIPrompt(row, args.dataset)
            try:
                prompt.check_conflicting_targets(tokenizer)
                assert '{premise}' in prompt.template
                assert '{hypothesis}' in prompt.template
            except (ValueError, AssertionError) as err_msg:
                writer.writerow({
                    'brand': args.model_name,
                    'template': prompt.template,
                    'targets': prompt.targets,
                    'error': str(err_msg)})
                out_file.flush()
                continue
            prompts.append(prompt)
            print(prompt, flush=True)
        if args.debug:  # only train on a small subset
            prompts = prompts[:1]

    for prompt in tqdm(prompts, desc='Prompts'):

        def prompt_and_tokenize(example: dict) -> dict:
            filled_template = prompt.template.format(
                premise=example['premise'],
                hypothesis=example['hypothesis'],
                mask=tokenizer.mask_token)
            return tokenizer(filled_template, truncation=True)

        # proc_train = train_set.map(prompt_and_tokenize)  # NOTE
            # remove_columns=['premise', 'hypothesis', 'idx'])
        proc_dev = dev_set.map(prompt_and_tokenize)
            # remove_columns=['premise', 'hypothesis'])  # idx, uid?

        if args.do_diagnosis:
            proc_diag = diagnostic_set.map(prompt_and_tokenize)  # type: ignore
        else:
            proc_diag = None

        for num_shots in tqdm(args.num_shots, desc='Num. Shots', disable=len(args.num_shots) == 1):
            result_table: list[dict] = []
            max_index = len(train_set) - num_shots
            start_index = random.randint(0, max_index)
            sample_indices = range(start_index, start_index + num_shots)
            k_shot_proc_train = train_set.select(sample_indices).map(prompt_and_tokenize)

            model = hf.AutoModelForMaskedLM.from_pretrained(args.model_name)
            setup_info: dict[str, Union[str, int, float]] = {
                'dataset': args.dataset,
                'brand': args.model_name,
                # 'm. param.': f'{model.num_parameters() / 1_000_000:.0f}',
                'template': prompt.template,
                'targets': prompt.targets,
                'template_name': prompt.template_name,
                'template_category': prompt.template_category,
                'target_category': prompt.target_category,
                'prompt_comment': prompt.comment,
                'num_shots': num_shots,
                'batch_size': args.train_batch_size,
                'starting_example_index': start_index,
                'seed': args.current_seed,
            }
            diagnoses = arrange_training(
                model, tokenizer, prompt, k_shot_proc_train, proc_dev, proc_diag)
            result_table += [setup_info | diagnosis for diagnosis in diagnoses]
            del model
            gc.collect()
            torch.cuda.empty_cache()

            # for row in result_table:
            #     for column in row.keys():
            #         if column not in manually_ordered_fieldnames:
            #             manually_ordered_fieldnames.append(column)
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
