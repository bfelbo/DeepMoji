""" Module import helper.
Modifies PATH in order to allow us to import the deepmoji directory.
"""
import sys
from os.path import abspath, dirname
sys.path.insert(0, dirname(dirname(abspath(__file__))))
