import unittest
import os
from knowledge_base.qa_system import TimeSeriesQA


class TestQASystem(unittest.TestCase):

    def setUp(self):
        """测试前设置"""
        self.qa_system = TimeSeriesQA(data_dir="test_data")

    def test_initialization(self):
        """测试系统初始化"""
        success = self.qa_system.initialize()
        self.assertTrue(success)

    def test_ask_question(self):
        """测试问答功能"""
        result = self.qa_system.ask("什么是ARIMA?")
        self.assertIn('question', result)
        self.assertIn('answer', result)


if __name__ == '__main__':
    unittest.main()