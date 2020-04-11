#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Smoke tests checking scripts in scripts/ run without raising a exception"""

import test_helper
import subprocess

from nose.plugins.attrib import attr


def test_smoke_calculate_coverages():
    subprocess.run(["python", "../scripts/calculate_coverages.py"], check=True)


def test_smoke_convert_all_datasets():
    subprocess.run(["python", "../scripts/convert_all_datasets.py"], check=True)


@attr("slow")
def test_smoke_download_weights():
    subprocess.run(["python", "../scripts/download_weights.py"],  check=True, input=b"y\n")


@attr("slow")
def test_smoke_finetune_dataset():
    subprocess.run(["python", "../scripts/finetune_dataset.py"], check=True)

# NB: Marked as slow because it relies on test_smoke_finetune_dataset
@attr("slow")
def test_smoke_analyze_results():
    # NOTE: Should be executed after test_smoke_finetune_dataset
    subprocess.run(["python", "../scripts/analyze_results.py"], check=True)

# NB: Marked as slow because it relies on test_smoke_finetune_dataset
@attr("slow")
def test_smoke_analyze_all_results():
    # NOTE: Should be executed after test_smoke_finetune_dataset, will fail otherwise
    #   due to missing result files.
    subprocess.run(["python", "../scripts/analyze_all_results.py"], check=True)
