from typing import List, NamedTuple
from collections import defaultdict
from pyctcdecode import build_ctcdecoder

import multiprocessing
import torch

from .char_text_encoder import CharTextEncoder
from hw_nv.utils import ROOT_PATH


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    def __init__(
            self,
            alphabet: List[str] = None,
            beam_size=2,
            use_lm=False
    ):
        super().__init__(alphabet)
        self.beam_size = beam_size
        self.EMPTY_TOK = "^"
        self.EMPTY_IND = 0
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.use_lm = use_lm

        if use_lm:
            KENLM_PATH = str(ROOT_PATH / "data/librispeech-lm/4-gram.arpa")

            # https://github.com/kensho-technologies/pyctcdecode
            # https://github.com/kensho-technologies/pyctcdecode/blob/main/tutorials/03_eval_performance.ipynb
            self.lm_ctcdecoder = build_ctcdecoder(
                labels=[""] + [c.upper() for c in self.alphabet],
                kenlm_model_path=KENLM_PATH,
                alpha=0.7,
                beta=3.0,
            )

    def ctc_decode(self, inds: List[int]) -> str:
        result = []
        last_char = self.EMPTY_TOK

        for ind in inds:
            char = self.ind2char[ind]
            if ind != self.EMPTY_IND and char != last_char:
                result.append(char)
            last_char = char

        return ''.join(result)
    
    def _extend_and_merge(self, frame, state):
        new_state = defaultdict(float)

        for next_char_index, next_char_prob in enumerate(frame):
            for (prefix, last_char), prefix_prob in state.items():
                next_char = self.ind2char[next_char_index]

                if next_char != last_char and next_char != self.EMPTY_TOK:
                    # prefix is string -> immutable -> it's ok
                    prefix += next_char

                new_state[(prefix, next_char)] += prefix_prob * next_char_prob
                last_char = next_char

        return new_state
    
    def _truncate(self, state, beam_size):
        sorted_stats_list = sorted(list(state.items()), key=lambda x: -x[1])
        return dict(sorted_stats_list[:beam_size])

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = None) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        hypos: List[Hypothesis] = []
        if beam_size is None:
            beam_size = self.beam_size

        # state[prefix, last_char] = prefix_prob
        state = {('', self.EMPTY_TOK): 1.0}

        for frame in probs[:probs_length]:
            state = self._extend_and_merge(frame, state)
            state = self._truncate(state, beam_size)
        
        for (prefix, last_char), prefix_prob in state.items():
            hypos.append(Hypothesis(text=prefix, prob=prefix_prob))
        
        return sorted(hypos, key=lambda x: x.prob, reverse=True)


    def ctc_lm_beam_search(self, probs, probs_length, beam_size: int = None) -> List[Hypothesis]:
        if beam_size is None:
            beam_size = self.beam_size
        
        if torch.is_tensor(probs) and probs.is_cuda:
            probs = probs.cpu()
        if torch.is_tensor(probs_length) and probs_length.is_cuda:
            probs_length = probs_length.cpu()

        probs = [prob[:length].numpy() for prob, length in zip(probs, probs_length)]
        
        with multiprocessing.get_context("fork").Pool() as pool:
            predicts = self.lm_ctcdecoder.decode_batch(pool, probs, beam_width=beam_size)
        
        predicts = [
            predict.lower().replace("'", "").replace("|", "").replace("??", "").strip()
            for predict in predicts
        ]
        
        return predicts