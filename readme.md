This repository accompanies our NAACL 2022 paper [“Do Prompt-Based Models Really Understand the Meaning of Their Prompts?”](https://arxiv.org/abs/2109.01247)

All figures in the paper have interactive versions pre-rendered in [this Colab](https://drive.google.com/file/d/1aZgereoKJYbUsADd3V6iorb-Xr06tFzZ/view?usp=sharing) where you can hover each scatter point to see their exact prompt, performance, and statistical test results without running any code.

## Usage

### Few-Shot Experiments
To replicate our results, for encoder-decoder models (e.g., T0 and T5 LM-Adapted), run:
```
python3 src/encoder_decoder.py \
    --brand bigscience/T0_3B \
    --save-dir runs/RTE_T0/ \
    --dataset rte \
    --prompt-path data/binary_NLI_prompts.csv \
    --experiment-name 'sec4' \
    --num-shots 4,8,16,32,64,128,256 \
    --epochs 30 \
    --train-batch-size 4 \
    --eval-batch-size 16 \
    --grad-accumulation 4 \
    --learning-rate 1e-4 \
    --production \
    --seed 1,2,3,4
```

For encoder-only models (e.g., BERT, RoBERTa, ALBERT), run:
```
python3 src/encoder.py \
    --model-name albert-xxlarge-v2 \
    --dataset rte \
    --save-dir runs/RTE_A2/ \
    --prompt-path data/binary_NLI_prompts.csv \
    --experiment-name 'sec4;encoder_only' \
    --num-shots 4,8,16,32,64,128,256 \
    --epochs 30 \
    --train-batch-size 4 \
    --eval-batch-size 16 \
    --grad-accumulation 4 \
    --learning-rate 1e-5 \
    --production \
    --seed 1,2,3,4
```

If you just want to run a quick test of dependencies, etc.:
```
python3 src/encoder.py \
    --model-name albert-xxlarge-v2 \
    --dataset rte \
    --save-dir runs/debug/ \
    --prompt-path data/binary_NLI_prompts.csv \
    --experiment-name 'encoder_only' \
    --num-shots 32 \
    --epochs 2 \
    --train-batch-size 4 \
    --eval-batch-size 16 \
    --grad-accumulation 4 \
    --learning-rate 1e-5 \
    --production \
    --seed 1
```

The `--dataset` argument can be any of `rte, cb, anli, mnli` (Recognizing Textual Entailment, CommitmentBank, AdversarialNLI, MultiNLI, respectively), but be sure to pass the right `--prompt-path`, i.e., a binary NLI prompt file for RTE and ternary prompt file for others. You can also pass `--do-diagnosis` to `encoder.py`, which produces fine-grained results of HANS (McCoy et al., 2019). We plan to write up an analysis on HANS in a future version of our paper. 

The `--experiment-name` argument specifies which groups of prompts to use from the the prompt CSV file.

Add `--fully-train` if you want to train on the entire training set in addition to few-shot settings.


### Zero-Shot Experiments
```
python3 src/encoder_decoder \
    --brand bigscience/T0 \
    --save-dir runs/zero-shot/  \
    --no-input-eos \
    --dataset rte \
    --prompt-path data/binary_NLI_prompts.csv \
    --experiment-name 'sec4' \
    --num-shots 0 \
    --production
```

### Target Word Experiments (Section 5)
```
python3 src/encoder.py.py \
    --model-name albert-xxlarge-v2 \
    --dataset rte \
    --save-dir runs/sec5/ \
    --prompt-path data/binary_NLI_prompts_permuted.csv \
    --experiment-name sec5 \
    --num-shots 32 \
    --production \
    --seeds 1,2,3,4
```

## Requirements
Activate your preferred virtual envrionment and then run `pip install -r requirements.txt`. You may need alternative commands to install CUDA 11 correctly; see PyTorch's official website for the latest instructions. If you want to replicate our exact results, you may need to use
```
torch==1.10.1+cu113
transformers==4.15.0
datasets==1.17.0
```

## Hyperparameters and Random Seeds
For encoder-only models, we follow Schick and Schütze (2021) and Le Scao and Rush (2021)’s recommendations and use a learning rate of 1e-5. For T5 and T0 models, we follow Raffel et al. (2020) and Sanh et al. (2022)’s recommendations and use a learning rate of 1e-4. We run several preliminary experiments with learning rates (3e^-4, 1e^-4, 5e-5, 1e-5) deviating from their recommendations and they perform worse, although our search is not exhaustive due to the high cost of running multiple prompts with multiple random seeds. 

Note that T5 and T0 are trained with the Adafactor optimizer in Mesh TensorFlow. Our implementation is in PyTorch, and we find that fine-tuning T5 with PyTorch's implementation of Adafactor yields substantially worse results than the usual choice of the AdamW optimizer. We corresponded with the authors of T5, who advised us that it might be due to the fact that PyTorch does not have the same learning rate scheduler implementation as TensorFlow's Adafactor does. They recommended us to simply use AdamW, which is what we did. This is somewhat unfortunate because Adafactor is much more memory efficient, which would have drastically reduced the compute resources required and thus enable more comprehensive experiments of the 11B models, which are currently limited to 0 shots and 16 shots only. 

Although most models seem to obtain the highest validation accuracy at very early epochs, we train all models to 30 epochs (20 epochs for 11B models) to be safe and select the checkpoint with the highest validation accuracy. 

All models use a batch size of 4 with 4 gradient accumulation steps for an effective batch size of 16.

Note that because we use a rank classification of single-token target words, decoding sampling methods (e.g., beam search, top-*k*, top-*p*) are unnecessary.

To replicate our exact results, use `--seeds 1,2,3,4`, which yields `starting_example_index` of `550,231,974,966` respectively. This is important for ensuring that all models trained under the same seed always see exactly the same training examples. See paper Section 3 for more details. If these seeds do not generate the same `starting_example_index` for you (which you can check in the output CSV files), you will have to manually specify the few-shot subset of training examples.

We follow Raffel et al. (2020) and add EOS tokens for input sequences, which yields higher few-shot performance compared to not adding EOS as done by Sanh et al. (2022). However, we omit EOS in the zero-shot setting, which exactly reproduces the results reported by Sanh et al. (2022). See T0's GitHub repository [readme](https://github.com/bigscience-workshop/t-zero/tree/master/examples) for more information.

### Compute Used
Each ALBERT 235M model is trained on a single Nvidia RTX3090. Their main experiments took approximately 192 GPU hours.

Each T5 LMA 770M model is trained on a single A6000. Their main experiments took approximately 48 GPU hours.

The 3B models are each trained by partitioning their layers over four RTX3090s. T5 and T0's main experiments took approximately 2,304 GPU hours in total. 

The 11B models are each trained on eight V100s (each with 32GB of memory). T5, T0, and T0++'s main experiments took approximately 1,728 GPU hours in total. (Due to their large GPU memory requirement, we were only able to complete their 0-shot and 16-shot experiments.)


## Citation
```
@misc{webson-pavlick-2021,
      title={Do Prompt-Based Models Really Understand the Meaning of their Prompts?}, 
      author={Albert Webson and Ellie Pavlick},
      year={2021},
      eprint={2109.01247},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```