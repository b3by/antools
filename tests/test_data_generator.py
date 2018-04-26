import unittest

from antools.utils import data_generator


class TestDataGeneratorUtils(unittest.TestCase):

    def test_next_window(self):
        test_signal = [list(range(200))]
        w = 10
        s = 5

        gen = data_generator.next_window(test_signal, w, s)
        first_win = next(gen)

        self.assertEqual(len(first_win), 10)
        self.assertEqual(first_win, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        second_win = next(gen)
        self.assertEqual(second_win, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

    def test_next_window_done(self):
        test_signal = [list(range(10))]
        w = 7
        s = 5

        gen = data_generator.next_window(test_signal, w, s)
        next(gen)

        self.assertRaises(StopIteration, next, gen)

    def test_next_window_multiple(self):
        test_s = [list(range(10)), list(range(10, 20)), list(range(20, 30))]
        w = 5
        s = 2

        gen = data_generator.next_window(test_s, w, s)
        win = next(gen)

        self.assertEqual(len(win), 15)
        self.assertEqual(
            win, [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24])

        win = next(gen)
        self.assertEqual(
            win, [2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26])


if __name__ == '__main__':
    unittest.main()
