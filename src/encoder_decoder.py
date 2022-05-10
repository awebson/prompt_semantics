import csv
import argparse
import random
import logging
import gc
from pathlib import Path
from typing import DefaultDict, Union, Optional

import transformers as hf
import datasets as hfd
import torch
import numpy as np
from tqdm import tqdm

from utils import table_columns


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--brand', type=str, default='bigscience/T0_3B')
parser.add_argument('-s', '--save-dir', type=Path, required=True)
parser.add_argument(
    '-d', '--dataset', type=str, required=True,
    choices=['rte', 'cb', 'anli', 'mnli'])
parser.add_argument('-pp', '--prompt-path', type=Path, required=True)
parser.add_argument('-ep', '--epochs', type=int, default=16)
parser.add_argument('-ns', '--num-shots', type=str, default='1,2,4,8,16,32,64,128,256')
parser.add_argument('--fully-train', action='store_true')
parser.add_argument('-tb', '--train-batch-size', type=int, default=8)
parser.add_argument('-eb', '--eval-batch-size', type=int, default=16)  # 64 max
parser.add_argument('-ga', '--grad-accumulation', type=int, default=1)
parser.add_argument('-fe', '--fixed-num-evals', type=int, default=0)
parser.add_argument('-ms', '--max-steps', type=int, default=-1)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-af', '--adafactor', action='store_true')
parser.add_argument('-mg', '--multi-gpu', action='store_true')
parser.add_argument('-mc', '--multi-cpu', type=int, default=0)
parser.add_argument('-exp', '--experiment-name', type=str, required=True)
parser.add_argument('--seeds', type=str, default='0')
parser.add_argument('-ft', '--no-input-eos', action='store_true')
parser.add_argument('--do-diagnosis', action='store_true')
# parser.add_argument('--subsample-subcase', type=int, default=100)  # only for HANS diagnosis
parser.add_argument('-ds', '--deepspeed-config', type=str, default=None)
parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
parser.add_argument('-wb', '--wandb-proj', type=str, default=None)
parser.add_argument('-wbn', '--wandb-name', type=str, default='')
parser.add_argument('-db', '--debug', action='store_true')
parser.add_argument('--production', action='store_true', help='suppress training metrics printing')
args = parser.parse_args()
args.seeds = list(map(int, args.seeds.split(',')))
args.num_shots = list(map(int, args.num_shots.split(',')))
args.save_dir.mkdir(parents=True, exist_ok=True)

if args.brand in ('albert-xxlarge-v2', 'roberta-large'):
    args.architecture = 'encoder'
    # from models import EncoderTrainer, MaskCollator
else:
    args.architecture = 'encoder_decoder'


# import sys
# logger = logging.getLogger(__name__)
# logging.basicConfig(#filename='stuff.log')#, level=logging.INFO)
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     handlers=[logging.StreamHandler(sys.stdout)]
# )

if args.wandb_proj:
    import wandb

hfd.set_progress_bar_enabled(False)
if args.production:
    hf.logging.set_verbosity(50)
    hfd.logging.set_verbosity(50)
    # import wandb
    # wandb.login()
else:
    from IPython import embed
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
        assert '{premise}' in self.template
        assert '{hypothesis}' in self.template

        if '\\n' in self.template:
            # print('Detected escaped newline in a template.')
            self.template = self.template.replace('\\n', '\n')

        if args.architecture == 'encoder':
            if '{mask}' not in self.template:
                self.template += ' {mask}'

        LM_targets = self.targets.rstrip().split(';')
        self.class_id_to_word: dict[int, str]
        if dataset in ('anli', 'mnli', 'cb'):
            self.ternary = True
            assert len(LM_targets) == 3
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
        self.word_id_to_class_id = {v: k for k, v in self.class_id_to_word_id.items()}
        return False

    def __str__(self):
        return self.template + ' -> ' + str(self.class_id_to_word)


def arrange_training(
        tokenizer: hf.AutoTokenizer,
        prompt: NLIPrompt,
        train_set: Optional[hfd.Dataset],
        dev_set: hfd.Dataset,
        diagnostic_set: Optional[hfd.Dataset] = None,
        ) -> list[dict]:
    if train_set is None:  # zero-shot
        adjusted_train_batch_size = 0
        eval_strategy = 'epoch'
        epochs = 0
        batched_eval_steps = None
    elif len(train_set) <= args.train_batch_size:
        adjusted_train_batch_size = len(train_set)  # a batch has the entire train set
        eval_strategy = 'epoch'
        # epochs = args.epochs * 2
        epochs = args.epochs
        batched_eval_steps = None
    else:
        adjusted_train_batch_size = args.train_batch_size
        eval_strategy = 'epoch'
        epochs = args.epochs
        batched_eval_steps = None

    if args.fixed_num_evals:
        eval_strategy = 'steps'
        if args.max_steps != -1:
            total_steps = args.max_steps
        else:
            total_steps = len(train_set) * epochs / adjusted_train_batch_size
        batched_eval_steps = total_steps / args.fixed_num_evals

    accuracy = hfd.load_metric('accuracy', keep_in_memory=True)
    label_word_ids = [wid for wid in prompt.class_id_to_word_id.values()]

    def eval_with_generate(eval_preds: tuple[np.ndarray, np.ndarray]) -> dict:
        preds, target_word_ids = eval_preds
        if isinstance(preds, tuple):
            logits = preds[0]  # [1] is probably final state
        else:
            logits = preds

        target_word_ids = target_word_ids[:, 0]
        # first_pred_word_ids = logits[:, 0]  # this is <BOS> or <pad>?
        second_pred_word_ids = logits[:, 1]
        # first_eval = accuracy.compute(predictions=first_pred_word_ids, references=target_word_ids)
        second_eval = accuracy.compute(predictions=second_pred_word_ids, references=target_word_ids)

        result = {
            'top1_acc': 0,
            'rank_acc': round(second_eval['accuracy'], 6),  # type: ignore
        }
        return result

    def eval_sans_generate(eval_preds: tuple[np.ndarray, np.ndarray]) -> dict:
        preds, target_word_ids = eval_preds
        if isinstance(preds, tuple):
            logits = preds[0]  # [1] is probably final state
        else:
            logits = preds
        target_word_ids = target_word_ids[:, 0]  # only need the first token for classification; second is </s>
        logits = logits[:, 0, :]  # batch * seq * vocab
        pred_word_ids = logits.argmax(axis=-1)
        naive_eval = accuracy.compute(predictions=pred_word_ids, references=target_word_ids)

        class_logits = logits[:, label_word_ids]  # (ent_id, neu_id, cont_id)
        pred_class = class_logits.argmax(axis=-1)
        int_class_ids = [prompt.word_id_to_class_id[wid] for wid in target_word_ids]
        rank_eval = accuracy.compute(predictions=pred_class, references=int_class_ids)

        result = {
            'top1_acc': round(naive_eval['accuracy'], 6),  # type: ignore
            'rank_acc': round(rank_eval['accuracy'], 6),  # type: ignore
        }
        return result

    train_args = hf.Seq2SeqTrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=False,
        evaluation_strategy=eval_strategy,
        eval_steps=batched_eval_steps,
        per_device_train_batch_size=adjusted_train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        num_train_epochs=epochs,
        max_steps=args.max_steps,
        # predict_with_generate=True,
        generation_max_length=2,
        logging_steps=8 if args.production else 1,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        remove_unused_columns=True,
        dataloader_num_workers=args.multi_cpu,
        disable_tqdm=args.production,
        log_level='info' if args.debug else 'warning',
        # report_to='wandb' if args.wandb_proj else None,
        # run_name=f'{prompt.template_name}_s{args.current_seed}',
        save_strategy='no',
        deepspeed=args.deepspeed_config,
        local_rank=args.local_rank,
        seed=args.current_seed
    )
    # pprint(train_args)
    print('Loading model...', flush=True)
    model = hf.AutoModelForSeq2SeqLM.from_pretrained(args.brand)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1 and args.deepspeed_config is None: #args.multi_gpu:
        model.parallelize()
        print(f'Parallelized over {num_gpus} GPUs.', flush=True)

    # tqdm.write(f'\n{prompt}')
    # tqdm.write(f"{train_set[0]['idx']} {train_set[0]['hypothesis']}")
    # tqdm.write(f"{train_set[-1]['idx']} {train_set[-1]['hypothesis']}")

    if args.adafactor:
        print('Using Adafactor')
        # # Constant LR
        # optimizer = hf.optimization.Adafactor(
        #     model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)
        # lr_scheduler = hf.optimization.AdafactorSchedule(optimizer, initial_lr=1e-3)

        # Adaptive LR
        optimizer = hf.optimization.Adafactor(
            model.parameters(), lr=None, scale_parameter=True, relative_step=True, warmup_init=True)
        lr_scheduler = hf.optimization.AdafactorSchedule(optimizer)
    else:
        # print('Using Adam')
        optimizer, lr_scheduler = None, None

    num_shot = len(train_set) if train_set is not None else 0
    reminder = {
        '_template_name': prompt.template_name,
        '_template': prompt.template,
        '_targets': prompt.targets,
        '_num_shot': num_shot,
        '_adjusted_train_batch_size': adjusted_train_batch_size,
        '_batched_eval_steps': batched_eval_steps,
    }

    if args.wandb_proj:
        num_shot = len(train_set) if train_set is not None else 0
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

    trainer = hf.Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        optimizers=(optimizer, lr_scheduler),
        # data_collator=collator,
        compute_metrics=eval_sans_generate,
        tokenizer=tokenizer,
    )
    # else:
    #     raise ValueError('Unknown architecture')

    if num_shot != 0:
        train_result = trainer.train()
    else:
        train_result = trainer.evaluate(dev_set)
    log_history = trainer.state.log_history

    if args.wandb_proj:
        wandb.finish()
    return log_history


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
    # if args.do_diagnosis:
    #     diagnostic_set = hfd.load_dataset('hans', split='validation')
    #     diagnostic_set = diagnostic_set.remove_columns(
    #         ['parse_premise', 'parse_hypothesis', 'binary_parse_premise', 'binary_parse_hypothesis'])
    #     table_columns += ['diag_avg', 'LE', 'SE', 'CE', 'LN', 'SN', 'CN']
    #     table_columns += HANS_subcases

    args.out_path = args.save_dir / f'{args.experiment_name}_s{args.current_seed}.csv'
    out_file = open(args.out_path, 'w')
    writer = csv.DictWriter(out_file, fieldnames=table_columns)
    writer.writeheader()
    # Load prompts
    tokenizer = hf.AutoTokenizer.from_pretrained(args.brand)
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
            except (ValueError, AssertionError) as err_msg:
                print(err_msg)
                writer.writerow({
                    'brand': args.brand,
                    'template': prompt.template,
                    'targets': prompt.targets,
                    'error': str(err_msg)})
                out_file.flush()
                continue
            prompts.append(prompt)
            print(prompt, flush=True)
        if args.debug:  # only train on a small subset
            prompts = prompts[:2]

    # Preprocessing and training
    for prompt in tqdm(prompts, desc='Prompts', disable=len(prompts) == 1):
        def prompt_and_tokenize(example: dict) -> dict:
            filled_template = prompt.template.format(
                premise=example['premise'],
                hypothesis=example['hypothesis'])
            model_input = tokenizer(filled_template, truncation=True, add_special_tokens=not args.no_input_eos)
            with tokenizer.as_target_tokenizer():
                target_word = prompt.class_id_to_word[example['label']]
                target_ids = tokenizer(target_word)['input_ids']
                target_ids = [
                    tid if tid != tokenizer.pad_token_id else -100
                    for tid in target_ids]
                model_input['labels'] = target_ids
                # del model_input['label']
            return model_input

        proc_dev = dev_set.map(prompt_and_tokenize, remove_columns='label')
        if args.do_diagnosis:
            proc_diag = diagnostic_set.map(prompt_and_tokenize)  # type: ignore
        else:
            proc_diag = None

        for num_shots in tqdm(args.num_shots, desc='Num. Shots', disable=len(args.num_shots) == 1):
            result_table: list[dict] = []
            if num_shots == 0:
                k_shot_proc_train = None
                start_index = None
            else:
                max_index = len(train_set) - num_shots
                start_index = random.randint(0, max_index)
                sample_indices = range(start_index, start_index + num_shots)
                k_shot_proc_train = train_set.select(sample_indices).map(
                    prompt_and_tokenize, remove_columns='label', )

            setup_info: dict[str, Union[str, int, float]] = {
                'dataset': args.dataset,
                'brand': args.brand,
                # 'm. param.': f'{model.num_parameters() / 1_000_000:.0f}',
                'template_name': prompt.template_name,
                'template_category': prompt.template_category,
                'target_category': prompt.target_category,
                'template': prompt.template,
                'targets': prompt.targets,
                'prompt_comment': prompt.comment,
                'num_shots': num_shots,
                'batch_size': args.train_batch_size,
                'starting_example_index': start_index,
                'seed': args.current_seed,
            }
            log_history = arrange_training(
                tokenizer, prompt, k_shot_proc_train, proc_dev, proc_diag)
            gc.collect()
            torch.cuda.empty_cache()
            result_table += [
                setup_info | {k: v for k, v in results.items() if k in table_columns}
                for results in log_history
            ]
            # Include all unspecified columns
            # result_table = [setup_info | result for result in log_history]
            # for row in result_table:
            #     for column in row.keys():
            #         if column not in table_columns:
            #             table_columns.append(column)
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
