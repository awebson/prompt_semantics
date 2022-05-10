import os
import sys


table_columns = [
    'template_name',
    'template_category',
    'target_category',
    'prompt_comment',
    'num_shots',
    'epoch',
    'step',
    # 'cumulative_examples',
    'eval_top1_acc',
    'eval_rank_acc',
    'eval_loss',
    'starting_example_index',
    'seed',
    'template',
    'targets',
    'dataset',
    'brand',
    'batch_size',
    'error',
]


HANS_subcases = (
    'ln_subject/object_swap',
    'ln_preposition',
    'ln_relative_clause',
    'le_relative_clause',
    'ln_passive',
    'le_passive',
    'ln_conjunction',
    'le_conjunction',
    'le_around_prepositional_phrase',
    'le_around_relative_clause',

    'sn_NP/S',
    'sn_PP_on_subject',
    'se_PP_on_obj',
    'sn_relative_clause_on_subject',
    'se_relative_clause_on_obj',
    'sn_past_participle',
    'sn_NP/Z',
    'se_conjunction',
    'se_adjective',
    'se_understood_object',

    'ce_after_since_clause',
    'cn_after_if_clause',
    'cn_embedded_under_if',
    'cn_embedded_under_verb',
    'ce_embedded_under_verb',
    'ce_embedded_under_since',
    'cn_disjunction',
    'ce_conjunction',
    'cn_adverb',
    'ce_adverb'
)


class Silence:
    def __init__(self, suppress_stdout=True, suppress_stderr=True):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        devnull = open(os.devnull, "w")
        if self.suppress_stdout:
            self._stdout = sys.stdout
            sys.stdout = devnull

        if self.suppress_stderr:
            self._stderr = sys.stderr
            sys.stderr = devnull

    def __exit__(self, *args):
        if self.suppress_stdout:
            sys.stdout = self._stdout
        if self.suppress_stderr:
            sys.stderr = self._stderr
