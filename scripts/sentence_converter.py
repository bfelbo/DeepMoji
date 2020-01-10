#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""load the pretrained deepmoji model and create a REPL prompting for sentences to
attribute emojis too. Prints out the top 5 emojis for a inputted sentence."""

import json
from typing import List

import numpy as np

from deepmoji.global_variables import VOCAB_PATH, PRETRAINED_PATH
from deepmoji.model_def import deepmoji_emojis
from deepmoji.sentence_tokenizer import SentenceTokenizer


EMOJI_MAP = {
    0: "😂",
    1: "😒",
    2: "😩",
    3: "😭",
    4: "😍",
    5: "😔",
    6: "👌",
    7: "😊",
    8: "❤",
    9: "😏",
    10: "😁",
    11: "🎶",
    12: "😳",
    13: "💯",
    14: "😴",
    15: "😌",
    16: "☺",
    17: "🙌",
    18: "💕",
    19: "😑",
    20: "😅",
    21: "🙏",
    22: "😕",
    23: "😘",
    24: "♥",
    25: "😐",
    26: "💁",
    27: "😞",
    28: "🙈",
    29: "😫",
    30: "✌",
    31: "😎",
    32: "😡",
    33: "👍",
    34: "😢",
    35: "😪",
    36: "😋",
    37: "😤",
    38: "✋",
    39: "😷",
    40: "👏",
    41: "👀",
    42: "🔫",
    43: "😣",
    44: "😈",
    45: "😓",
    46: "💔",
    47: "♡",
    48: "🎧",
    49: "🙊",
    50: "😉",
    51: "💀",
    52: "😖",
    53: "😄",
    54: "😜",
    55: "😠",
    56: "🙅",
    57: "💪",
    58: "👊",
    59: "💜",
    60: "💖",
    61: "💙",
    62: "😬",
    63: "✨",
}


def get_vocabulary():
    with open(VOCAB_PATH, "r") as f:
        return json.load(f)


def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]


def get_top_emojis(st, deepmoji_model, sentence: str) -> List[str]:
    tokenized, _, _ = st.tokenize_sentences([sentence])
    prob = deepmoji_model.predict(tokenized)
    for i, t_prob in enumerate(prob):
        return list(
            [EMOJI_MAP[emoji_index] for emoji_index in list(top_elements(t_prob, 5))]
        )


def main():
    sentence_tokenizer = SentenceTokenizer(get_vocabulary(), 30)
    deepmoji_model = deepmoji_emojis(
        maxlen=30,
        weight_path=PRETRAINED_PATH,
    )
    while True:
        sentence = input("enter sentence:").strip()
        if sentence:
            top_emojis = get_top_emojis(sentence_tokenizer, deepmoji_model, sentence)
            print("top related emojis:", top_emojis)


if __name__ == "__main__":
    main()
