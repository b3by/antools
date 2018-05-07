import unittest
import os

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

    def test_get_win(self):
        test_file = os.path.join(os.path.dirname(__file__),
                                 'test_signal.csv')
        test_c = [319, 644,
                  788, 1125,
                  1287, 1598,
                  1745, 2086,
                  2245, 2554,
                  2690, 3009,
                  3121, 3430,
                  3565, 3852,
                  3961, 4257,
                  4343, 4684]
        crds = list(zip(test_c, test_c[1:]))

        df, _ = data_generator.get_win(test_file, crds, ['arm'], 40, 1)
        self.assertEqual(4784, len(df))

        df, _ = data_generator.get_win(test_file, crds, ['arm'], 40, 2)
        self.assertEqual(2392, len(df))

        df, _ = data_generator.get_win(test_file, crds, ['arm'], 40, 3)
        self.assertEqual(1595, len(df))

        df, _ = data_generator.get_win(test_file, crds, ['arm'], 50, 7)
        self.assertEqual(682, len(df))


if __name__ == '__main__':
    unittest.main()
