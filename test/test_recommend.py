#!/usr/bin/env python
# -*- coding:utf-8
import unittest
from recommend import Calc

class TestSample(unittest.TestCase):
  def test_sample(self):
    calc = Calc()
    self.assertEqual(15, calc.add(10, 5))
