import unittest
import os

from antools.utils import data_generator
import numpy as np


class TestDataGeneratorUtils(unittest.TestCase):

    def test_next_window(self):
        test_signal = [list(range(1000))]
        w = 10
        s = 5

        gen = data_generator.next_window(test_signal, w, s)

        for idx, win in enumerate(gen):
            self.assertEqual(10, len(win))
            self.assertCountEqual(list(range((idx * s), (idx * s) + w)), win)

    def test_next_window_count(self):
        test_signal = [list(range(1000))]
        w = 10
        s = 5

        gen = data_generator.next_window(test_signal, w, s)
        c = 0

        for win in gen:
            c += 1

        self.assertEqual((1000 - 10) / 5 + 1, c)

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

    def test_get_files(self):
        base = os.path.dirname(__file__)
        test_dataset = os.path.join(base, 'data')
        test_coordinate_file = os.path.join(base, 'test_coordinates.csv')

        f1 = os.path.join(base, 'data/ex1/ex1.real.csv')
        f2 = os.path.join(base, 'data/ex1/ex1.mock.csv')

        ll = data_generator.get_files(test_dataset, test_coordinate_file)

        self.assertEqual(2, len(ll))
        self.assertEqual(2, len(ll[1][1]))

        self.assertEqual(f1, ll[0][0])
        self.assertEqual(f2, ll[1][0])

        self.assertCountEqual([(319, 644),
                               (788, 1125),
                               (1287, 1598),
                               (1745, 2086),
                               (2245, 2554),
                               (2690, 3009),
                               (3121, 3430),
                               (3565, 3852),
                               (3961, 4257),
                               (4343, 4684)], ll[0][1])
        self.assertCountEqual([(20, 40), (60, 80)], ll[1][1])

    def test_get_files_with_exercise(self):
        base = os.path.dirname(__file__)
        test_dataset = os.path.join(base, 'data')
        test_coordinate_file = os.path.join(base, 'test_coordinates.csv')

        f1 = os.path.join(base, 'data/ex1/ex1.real.csv')
        f2 = os.path.join(base, 'data/ex1/ex1.mock.csv')

        ll = data_generator.get_files(test_dataset, test_coordinate_file,
                                      ['ex1'])

        self.assertEqual(2, len(ll))
        self.assertEqual(2, len(ll[1][1]))

        self.assertEqual(f1, ll[0][0])
        self.assertEqual(f2, ll[1][0])

        self.assertCountEqual([(319, 644),
                               (788, 1125),
                               (1287, 1598),
                               (1745, 2086),
                               (2245, 2554),
                               (2690, 3009),
                               (3121, 3430),
                               (3565, 3852),
                               (3961, 4257),
                               (4343, 4684)], ll[0][1])
        self.assertCountEqual([(20, 40), (60, 80)], ll[1][1])

    def test_get_win_count(self):
        base = os.path.dirname(__file__)
        test_dataset = os.path.join(base, 'data')
        test_coordinate_file = os.path.join(base, 'test_coordinates.csv')

        ff = data_generator.get_files(test_dataset,
                                      test_coordinate_file)[1]

        df, _ = data_generator.get_win(ff[0], ff[1], ['target1'], 10, 1)
        self.assertEqual(int(((100 - 10) / 1) + 1), df.shape[0])

        df, _ = data_generator.get_win(ff[0], ff[1], ['target1'], 20, 1)
        self.assertEqual(int(((100 - 20) / 1) + 1), df.shape[0])

        df, _ = data_generator.get_win(ff[0], ff[1], ['target1'], 20, 5)
        self.assertEqual(int(((100 - 20) / 5) + 1), df.shape[0])

        df, _ = data_generator.get_win(ff[0], ff[1], ['target1'], 18, 7)
        self.assertEqual(int(((100 - 18) / 7) + 1), df.shape[0])

    def test_get_win_mock(self):
        base = os.path.dirname(__file__)
        test_dataset = os.path.join(base, 'data')
        test_coordinate_file = os.path.join(base, 'test_coordinates.csv')

        ff = data_generator.get_files(test_dataset,
                                      test_coordinate_file)[1]

        df, freqs = data_generator.get_win(ff[0], ff[1],
                                           ['target1'], 10, 1)

        self.assertEqual(10 * 6 + 1, df.shape[1])

        self.assertEqual(29, freqs[0])
        self.assertEqual(44, freqs[1])
        self.assertEqual(18, freqs[2])

        synth = df.values.tolist()
        tg = []

        for x in range(1, 7):
            tg += [x, x + 1] + [x] * 8

        tg.append(0)

        self.assertCountEqual(tg, synth[0])

    def test_get_win_binary_mock_stride_one(self):
        base = os.path.dirname(__file__)
        test_dataset = os.path.join(base, 'data')
        test_coordinate_file = os.path.join(base, 'test_coordinates.csv')

        ff = data_generator.get_files(test_dataset,
                                      test_coordinate_file)[1]

        df, freqs = data_generator.get_win(ff[0], ff[1],
                                           ['target1'], 10, 1, binary=True)

        self.assertEqual(91, df.shape[0])
        self.assertCountEqual([0] * 11, df['label'].values.tolist()[:11])
        self.assertCountEqual([1] * 29, df['label'].values.tolist()[11:40])
        self.assertCountEqual([0] * 11, df['label'].values.tolist()[40:51])
        self.assertCountEqual([1] * 29, df['label'].values.tolist()[51:80])
        self.assertCountEqual([0] * 11, df['label'].values.tolist()[80:91])

    def test_get_win_binary_mock_other_strides(self):
        base = os.path.dirname(__file__)
        test_dataset = os.path.join(base, 'data')
        test_coordinate_file = os.path.join(base, 'test_coordinates.csv')

        ff = data_generator.get_files(test_dataset,
                                      test_coordinate_file)[1]

        df, freqs = data_generator.get_win(ff[0], ff[1],
                                           ['target1'], 3, 2, binary=True)

        self.assertEqual(49, df.shape[0])
        self.assertCountEqual([0] * 4, df['label'].values.tolist()[:4])

    def test_get_win_normalize_minmax(self):
        base = os.path.dirname(__file__)
        test_dataset = os.path.join(base, 'data')
        test_coordinate_file = os.path.join(base, 'test_coordinates.csv')

        ff = data_generator.get_files(test_dataset,
                                      test_coordinate_file)[1]

        df, freqs = data_generator.get_win(ff[0], ff[1],
                                           ['target1'], 10, 1,
                                           normalize='minmax')

        first_win = df.loc[0].tolist()
        tg = []

        for x in range(1, 7):
            tg += [0.0, 1.0] + [0.0] * 8

        tg.append(0)

        self.assertEqual(61, len(first_win))
        self.assertCountEqual(tg, first_win)
        self.assertEqual(0.0, first_win[0])
        self.assertEqual(1.0, first_win[1])

    def test_get_win_normalize_zscore(self):
        base = os.path.dirname(__file__)
        test_dataset = os.path.join(base, 'data')
        test_coordinate_file = os.path.join(base, 'test_coordinates.csv')

        ff = data_generator.get_files(test_dataset,
                                      test_coordinate_file)[1]

        df, freqs = data_generator.get_win(ff[0], ff[1],
                                           ['target1'], 10, 1,
                                           normalize='zscore')

        first_win = df.loc[0].tolist()

        tg = []

        for x in range(1, 7):
            tg += [-0.100503781525, 9.949874371] + [-0.100503781525] * 8
        tg.append(0.0)

        self.assertEqual(61, len(first_win))
        self.assertTrue(np.allclose(tg, first_win, rtol=1e-04, atol=1e-08))


if __name__ == '__main__':
    unittest.main()
