import unittest
from web_dynamic.OptiDash import nsga2



class TestNSGA2(unittest.TestCase):
    def test_function_runs_without_errors(self):
        # Test if the function runs without raising any exceptions
        self.assertIsNone(nsga2([]))



if __name__ == '__main__':
    unittest.main()