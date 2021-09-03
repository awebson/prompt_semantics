This repository accompanies our paper [“Do Prompt-Based Models Really Understand the Meaning of Their Prompts?”](./Webson_and_Pavlick_2021.pdf)

## Usage

To replicate our results in Section 4, run:
```
python3 prompt_tune.py \
    --save-dir ../runs/prompt_tuned_sec4/ \
    --prompt-path ../data/binary_NLI_prompts.csv \
    --experiment-name sec4 \
    --few-shots 3,5,10,20,30,50,100,250 \
    --production \
    --seeds 1
```
Add `--fully-train` if you want to train on the entire training set in addition to few-shot settings.

To replicate Section 5, run:
```
python3 prompt_tune.py \
    --save-dir ../runs/prompt_tuned_sec5/ \
    --prompt-path ../data/binary_NLI_prompts_permuted.csv \
    --experiment-name sec5 \
    --few-shots 3,5,10,20,30,50,100,250 \
    --production \
    --seeds 1
```

To get a fine-tuning baseline (Figure 1):
```
python3 fine_tune.py \
    --save-dir ../runs/fine_tune/ \
    --epochs 5 \
    --few-shots 3,5,10,20,30,50,100,250 \
    --fully-train \
    --production \
    --seeds 1
```

To replicate our exact results, use `--seeds 1,2,3,4,5,6,7,8`, which yields `starting_example_index` of `550,231,974,966,1046,2350,1326,928` respectively. This is important for ensuring that all models trained under the same seed always see exactly the same training examples. See paper Section 3 for more details.

If these seeds do not generate the same `starting_example_index` for you (which you can check in the output CSV files), you will have to manually specify the few-shot subset of training examples. I plan to add an argparse argument for this to make it easy.

All other hyperparameters are the same as the argparse default.

## Miscellaneous Notes

You might notice that the code and output files are set up to produce a fine-grained analysis of HANS ([McCoy et al., 2019](https://aclanthology.org/P19-1334/)). We actually run all of our main experiments on HANS as well and got similar results, which we plan to write up in a future version of our paper. Meanwhile, if you’re curious, feel free to add `--do-diagnosis` which will report the results on HANS.

## Requirements

Python 3.9.

3.7 should mostly work too. You’d have to just replace the new [built-in type hints](https://www.python.org/dev/peps/pep-0585/) and dictionary union operators with their older equivalents.

Activate your preferred virtual envrionment and then run `pip install -r requirements.txt`. If you want to replicate our exact results, use
```
torch==1.9.0+cu111
transformers==4.9.2
datasets==1.11.0
```
