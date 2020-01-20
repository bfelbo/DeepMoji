#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Some basic smoke tests ensure that the python scripts example/ at least run with
raising a exception."""

import test_helper
import subprocess


def test_smoke_create_twitter_vocab():
    subprocess.run(["python", "../examples/create_twitter_vocab.py"], check=True)


def test_smoke_dataset_split():
    subprocess.run(["python", "../examples/dataset_split.py"], check=True)


def test_smoke_encode_texts():
    subprocess.run(["python", "../examples/encode_texts.py"], check=True)


def test_smoke_finetune_insults_chain_thaw():
    subprocess.run(["python", "../examples/finetune_insults_chain-thaw.py"], check=True)


def test_smoke_finetune_semeval_class_avg_f1():
    subprocess.run(["python", "../examples/finetune_semeval_class-avg_f1.py"], check=True)


def test_smoke_finetune_youtube_last():
    subprocess.run(["python", "../examples/finetune_youtube_last.py"], check=True)


def test_smoke_imdb_from_scratch():
    subprocess.run(["python", "../examples/imdb_from_scratch.py"], check=True)


def test_smoke_score_texts_emojis():
    subprocess.run(["python", "../examples/score_texts_emojis.py"], check=True)


def test_smoke_tokenize_dataset():
    subprocess.run(["python", "../examples/tokenize_dataset.py"], check=True)


def test_smoke_vocab_extension():
    subprocess.run(["python", "../examples/vocab_extension.py"], check=True)
