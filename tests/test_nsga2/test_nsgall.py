import unittest
from web_dynamic.OptiDash import nsga2, nsgaa2, non_dominates
from web_dynamic.OptiDash import dominates1, create_objective_function
from web_dynamic.OptiDash import non_dominates1, dominates
from web_dynamic.OptiDash import check, makedub
import uuid


class TestNSGA2(unittest.TestCase):
    def test_function_runs_without_errors_nsga2(self):
        # Test if the function runs without raising any exceptions
        task_id = str(uuid.uuid4())
        table_data = [['x1', '0', '1'], ['x2', '0', '3']]
        table_data1 = ['x1', '1+x2-x1**2']
        table_data2 = ['Minimization', 'Minimization']

        self.assertIsNone(
            nsga2(table_data, table_data1, table_data2, task_id))

    def test_function_runs_without_errors_nsgaa2(self):
        # Test if the function runs without raising any exceptions
        task_id = str(uuid.uuid4())
        table_data = [['x1', '0', '1'], ['x2', '0', '3']]
        table_data1 = ['x1', '1+x2-x1**2', 'x1**2']
        table_data2 = ['Minimization', 'Minimization', 'Minimization']
        self.assertIsNone(
            nsgaa2(table_data, table_data1, table_data2, task_id))

    def test_complex_formula(self):
        variables = ['x1', 'x2', 'x3']
        formula = '(x1**2) + (x2**2) - (x3**2) + (2*x1*x2) - (x1*x3)'
        objective_function = create_objective_function(variables, formula)
        # Test with values
        self.assertEqual(objective_function([1, 2, 3]), -3)
        self.assertEqual(objective_function([3, 4, 5]), 9)

    def test_multiple_variables(self):
        variables = ['x1', 'x2']
        formula = 'x1 * x2 + x2 - x1'
        objective_function = create_objective_function(variables, formula)
        # Test with values
        self.assertEqual(objective_function([2, 3]), 7)
        self.assertEqual(objective_function([1, 1]), 1)

    def test_single_variable(self):
        variables = ['x1']
        formula = 'x1 + 2'
        objective_function = create_objective_function(variables, formula)
        # Test with a value
        self.assertEqual(objective_function([3]), 5)

    def test_dominates1(self):
        self.assertTrue(
            dominates1([1, 2], [2, 3],
                       ["Minimization", "Minimization"]))
        self.assertFalse(
            dominates1([3, 2], [2, 1],
                       ["Minimization", "Minimization"]))
        self.assertTrue(
            dominates1([3, 2], [1, 0],
                       ["Maximization", "Maximization"]))
        self.assertFalse(
            dominates1([1, 2], [3, 4],
                       ["Maximization", "Maximization"]))
        self.assertTrue(
            dominates1([1, 3], [4, 2],
                       ["Minimization", "Maximization"]))
        self.assertFalse(
            dominates1([4, 3], [1, 2],
                       ["Minimization", "Maximization"]))
        self.assertTrue(
            dominates1([3, 1], [2, 4],
                       ["Maximization", "Minimization"]))
        self.assertFalse(
            dominates1([1, 3], [4, 2],
                       ["Maximization", "Minimization"]))

    def test_non_dominates1(self):
        self.assertTrue(
            non_dominates1([1, 2], [2, 3],
                           ["Minimization", "Minimization"]))
        self.assertFalse(
            non_dominates1([3, 2], [2, 1],
                           ["Minimization", "Minimization"]))
        self.assertTrue(
            non_dominates1([3, 2], [1, 0],
                           ["Maximization", "Maximization"]))
        self.assertFalse(
            non_dominates1([1, 2], [3, 4],
                           ["Maximization", "Maximization"]))
        self.assertTrue(
            non_dominates1([1, 3], [4, 2],
                           ["Minimization", "Maximization"]))
        self.assertFalse(
            non_dominates1([4, 3], [1, 2],
                           ["Minimization", "Maximization"]))
        self.assertTrue(
            non_dominates1([3, 1], [2, 4],
                           ["Maximization", "Minimization"]))
        self.assertFalse(
            non_dominates1([1, 3], [4, 2],
                           ["Maximization", "Minimization"]))

    def test_dominates(self):
        self.assertTrue(
            dominates([1, 2], [2, 3], [0, 1],
                      ["Minimization", "Minimization", "Minimization"]))
        self.assertFalse(
            dominates([3, 2], [2, 1], [4, 3],
                      ["Minimization", "Minimization", "Minimization"]))
        self.assertTrue(
            dominates([3, 2], [1, 0], [2, 1],
                      ["Maximization", "Maximization", "Maximization"]))
        self.assertFalse(
            dominates([1, 2], [3, 4], [0, 1],
                      ["Maximization", "Maximization", "Maximization"]))
        self.assertTrue(
            dominates([3, 2], [1, 3], [4, 5],
                      ["Maximization", "Minimization", "Minimization"]))
        self.assertFalse(
            dominates([1, 2], [3, 4], [0, 1],
                      ["Maximization", "Minimization", "Minimization"]))
        self.assertFalse(
            dominates([1, 2], [3, 4], [5, 6],
                      ["Minimization", "Maximization", "Maximization"]))
        self.assertFalse(
            dominates([3, 2], [1, 0], [2, 1],
                      ["Minimization", "Maximization", "Maximization"]))
        self.assertTrue(
            dominates([3, 2], [1, 0], [2, 3],
                      ["Maximization", "Maximization", "Minimization"]))
        self.assertFalse(
            dominates([1, 2], [3, 4], [0, 1],
                      ["Maximization", "Maximization", "Minimization"]))
        self.assertFalse(
            dominates([1, 2], [3, 4], [5, 6],
                      ["Minimization", "Minimization", "Maximization"]))
        self.assertFalse(
            dominates([3, 2], [1, 0], [2, 1],
                      ["Minimization", "Minimization", "Maximization"]))
        self.assertFalse(
            dominates([3, 2], [1, 0], [2, 3],
                      ["Maximization", "Minimization", "Maximization"]))
        self.assertFalse(
            dominates([1, 2], [3, 4], [0, 1],
                      ["Maximization", "Minimization", "Maximization"]))
        self.assertFalse(
            dominates([1, 2], [3, 4], [5, 6],
                      ["Minimization", "Maximization", "Minimization"]))
        self.assertFalse(
            dominates([3, 2], [1, 0], [2, 1],
                      ["Minimization", "Maximization", "Minimization"]))

    def test_non_dominates(self):
        self.assertTrue(
            non_dominates([3, 2], [2, 3], [0, 1],
                          ["Minimization", "Minimization", "Minimization"]))
        self.assertFalse(
            non_dominates([3, 2], [2, 1], [4, 3],
                          ["Minimization", "Minimization", "Minimization"]))
        self.assertTrue(
            non_dominates([3, 2], [1, 0], [2, 1],
                          ["Maximization", "Maximization", "Maximization"]))
        self.assertFalse(
            non_dominates([1, 2], [3, 4], [0, 1],
                          ["Maximization", "Maximization", "Maximization"]))
        self.assertTrue(
            non_dominates([3, 2], [1, 3], [4, 5],
                          ["Maximization", "Minimization", "Minimization"]))
        self.assertTrue(
            non_dominates([1, 2], [3, 4], [0, 1],
                          ["Maximization", "Minimization", "Minimization"]))
        self.assertTrue(
            non_dominates([1, 2], [3, 4], [5, 6],
                          ["Minimization", "Maximization", "Maximization"]))
        self.assertTrue(
            non_dominates([3, 2], [1, 0], [2, 1],
                          ["Minimization", "Maximization", "Maximization"]))
        self.assertTrue(
            non_dominates([3, 2], [1, 0], [2, 3],
                          ["Maximization", "Maximization", "Minimization"]))
        self.assertTrue(
            non_dominates([1, 2], [3, 4], [0, 1],
                          ["Maximization", "Maximization", "Minimization"]))
        self.assertTrue(
            non_dominates([1, 2], [3, 4], [5, 6],
                          ["Minimization", "Minimization", "Maximization"]))
        self.assertTrue(
            non_dominates([3, 2], [1, 0], [2, 1],
                          ["Minimization", "Minimization", "Maximization"]))
        self.assertTrue(
            non_dominates([3, 2], [1, 0], [2, 3],
                          ["Maximization", "Minimization", "Maximization"]))
        self.assertTrue(
            non_dominates([1, 2], [3, 4], [0, 1],
                          ["Maximization", "Minimization", "Maximization"]))
        self.assertTrue(
            non_dominates([1, 2], [3, 4], [5, 6],
                          ["Minimization", "Maximization", "Minimization"]))
        self.assertTrue(
            non_dominates([3, 2], [1, 0], [2, 1],
                          ["Minimization", "Maximization", "Minimization"]))

    def test_check(self):
        # Testing the function with a list
        # containing all the same elements
        self.assertTrue(check([1, 1, 1, 1], 1))
        # Testing the function with a list
        # containing different elements
        self.assertFalse(check([1, 2, 3, 4], 1))
        # Testing the function with an empty list
        self.assertFalse(check([], 1))
        # Testing the function with a list
        # where the value is not present
        self.assertFalse(check([1, 2, 3, 4], 5))
        # Testing the function with a list where
        # the value is present but not all elements are the same
        self.assertFalse(check([1, 1, 2, 1], 1))

    def test_makedub(self):
        # Testing the function with a list of integers
        self.assertEqual(makedub([1, 2, 3]), [[1], [2], [3]])
        # Testing the function with a list of strings
        self.assertEqual(makedub(
            ['a', 'b', 'c']),
            [['a'], ['b'], ['c']])
        # Testing the function with an empty list
        self.assertEqual(makedub([]), [])
        # Testing the function with a list
        # containing a single element
        self.assertEqual(makedub([5]), [[5]])
        # Testing the function with a list containing
        # multiple elements of the same value
        self.assertEqual(makedub([1, 1, 1]), [[1], [1], [1]])


if __name__ == '__main__':
    unittest.main()
