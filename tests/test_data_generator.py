import unittest
import os
import shutil
import types

from antools.utils import data_generator
import pandas as pd
import numpy as np


class TestDataGeneratorUtils(unittest.TestCase):

    base = os.path.dirname(__file__)
    test_dataset = os.path.join(os.path.dirname(__file__), 'data')
    gen_path = os.path.join(os.path.dirname(__file__), 'data', 'generated')
    test_coordinate_file = os.path.join(os.path.dirname(__file__),
                                        'test_coordinates.csv')

    def setUp(self):
        os.makedirs(self.gen_path)

    def tearDown(self):
        if os.path.exists(self.gen_path):
            shutil.rmtree(self.gen_path)

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
        f1 = os.path.join(self.base, 'data/ex1/ex1.0.real.0.csv')
        f2 = os.path.join(self.base, 'data/ex1/ex1.0.mock.0.csv')

        ll = data_generator.get_files(self.test_dataset,
                                      self.test_coordinate_file)

        self.assertEqual(3, len(ll))
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
        f1 = os.path.join(self.base, 'data/ex1/ex1.0.real.0.csv')
        f2 = os.path.join(self.base, 'data/ex1/ex1.0.mock.0.csv')

        ll = data_generator.get_files(self.test_dataset,
                                      self.test_coordinate_file, ['ex1'])

        self.assertEqual(3, len(ll))
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
        ff = data_generator.get_files(self.test_dataset,
                                      self.test_coordinate_file)[1]

        g = data_generator.get_win(ff[0], ff[1], ['target1'], 10, 1,
                                   self.gen_path)
        self.assertEqual(g, os.path.join(self.gen_path, 'ex1.0.mock.0.csv'))
        df = pd.read_csv(g)
        self.assertEqual(int(((100 - 10) / 1) + 1), df.shape[0])

        g = data_generator.get_win(ff[0], ff[1], ['target1'], 20, 1,
                                   self.gen_path)
        df = pd.read_csv(g)
        self.assertEqual(int(((100 - 20) / 1) + 1), df.shape[0])

        g = data_generator.get_win(ff[0], ff[1], ['target1'], 20, 5,
                                   self.gen_path)
        df = pd.read_csv(g)
        self.assertEqual(int(((100 - 20) / 5) + 1), df.shape[0])

        g = data_generator.get_win(ff[0], ff[1], ['target1'], 18, 7,
                                   self.gen_path)
        df = pd.read_csv(g)
        self.assertEqual(int(((100 - 18) / 7) + 1), df.shape[0])

    def test_get_win_mock(self):
        ff = data_generator.get_files(self.test_dataset,
                                      self.test_coordinate_file)[1]

        g = data_generator.get_win(ff[0], ff[1], ['target1'], 10, 1,
                                   self.gen_path)

        df = pd.read_csv(g)

        self.assertEqual(10 * 6 + 1, df.shape[1])

        synth = df.values.tolist()
        tg = []

        for x in range(1, 7):
            tg += [x, x + 1] + [x] * 8

        tg.append(0)

        self.assertCountEqual(tg, synth[0])

    def test_get_win_binary_mock_stride_one(self):
        ff = data_generator.get_files(self.test_dataset,
                                      self.test_coordinate_file)[1]

        g = data_generator.get_win(ff[0], ff[1], ['target1'], 10, 1,
                                   self.gen_path)

        df = pd.read_csv(g)

        self.assertEqual(91, df.shape[0])
        self.assertCountEqual([0] * 11, df['label'].values.tolist()[:11])
        self.assertCountEqual([1] * 29, df['label'].values.tolist()[11:40])
        self.assertCountEqual([0] * 11, df['label'].values.tolist()[40:51])
        self.assertCountEqual([1] * 29, df['label'].values.tolist()[51:80])
        self.assertCountEqual([0] * 11, df['label'].values.tolist()[80:91])

    def test_get_win_binary_mock_other_strides(self):
        ff = data_generator.get_files(self.test_dataset,
                                      self.test_coordinate_file)[1]

        g = data_generator.get_win(ff[0], ff[1], ['target1'], 3, 2,
                                   self.gen_path)

        df = pd.read_csv(g)

        self.assertEqual(49, df.shape[0])
        self.assertCountEqual([0] * 4, df['label'].values.tolist()[:4])

    def test_get_win_normalize_minmax(self):
        ff = data_generator.get_files(self.test_dataset,
                                      self.test_coordinate_file)[1]

        g = data_generator.get_win(ff[0], ff[1], ['target1'], 10, 1,
                                   self.gen_path, normalize='minmax')

        df = pd.read_csv(g)
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
        ff = data_generator.get_files(self.test_dataset,
                                      self.test_coordinate_file)[1]

        g = data_generator.get_win(ff[0], ff[1], ['target1'], 10, 1,
                                   self.gen_path, normalize='zscore')

        df = pd.read_csv(g)
        first_win = df.loc[0].tolist()

        tg = []

        for x in range(1, 7):
            tg += [-0.100503781525, 9.949874371] + [-0.100503781525] * 8
        tg.append(0.0)

        self.assertEqual(61, len(first_win))
        self.assertTrue(np.allclose(tg, first_win, rtol=1e-04, atol=1e-08))

    def test_exception_in_window_generation(self):
        ff = data_generator.get_files(self.test_dataset,
                                      self.test_coordinate_file)[1]

        g = data_generator.get_win(ff[0], ff[1], ['target1'], 10, 1,
                                   'faulty_loc')

        self.assertIsNone(g)

    def test_get_win_no_sensors(self):
        ff = data_generator.get_files(self.test_dataset,
                                      self.test_coordinate_file)[2]

        g = data_generator.get_win(ff[0], ff[1], ['missing'], 10, 1,
                                   self.gen_path)

        df = pd.read_csv(g)
        self.assertEqual(61, len(df.loc[0].tolist()))

    def test_concatenate_and_save(self):
        data_generator.concatenate_and_save(
            os.path.join(self.test_dataset, 'aggregated'),
            os.path.join(self.gen_path, 'full.csv'))

        full = pd.read_csv(os.path.join(self.gen_path, 'full.csv'))

        self.assertEqual(4823 * 2, full.shape[0])

    def test_generate_input(self):
        tr, ts = data_generator.generate_input(
            self.test_dataset,
            os.path.join(self.gen_path, 'train.csv'),
            os.path.join(self.gen_path, 'test.csv'),
            self.test_coordinate_file,
            ['target1'], 10, 1)

        self.assertEqual([], tr)
        self.assertEqual([0], ts)

        self.assertTrue(os.path.exists(
            os.path.join(self.gen_path, 'test.csv')))

        df = pd.read_csv(os.path.join(self.gen_path, 'test.csv'))

        self.assertEqual(61, df.shape[1])
        self.assertEqual(182, df.shape[0])

    def test_generate_input_with_max_files(self):
        tr, ts = data_generator.generate_input(
            self.test_dataset,
            os.path.join(self.gen_path, 'train.csv'),
            os.path.join(self.gen_path, 'test.csv'),
            self.test_coordinate_file,
            ['target1'], 10, 1, max_files=2)

        df = pd.read_csv(os.path.join(self.gen_path, 'test.csv'))

        self.assertEqual(61, df.shape[1])
        self.assertEqual(91, df.shape[0])

    def test_generate_dataset(self):
        flg = types.SimpleNamespace()
        flg.dataset_location = self.test_dataset
        flg.train = os.path.join(self.gen_path, 'train.csv')
        flg.test = os.path.join(self.gen_path, 'test.csv')
        flg.coordinates = self.test_coordinate_file
        flg.sensors = ['target1']
        flg.window_size = 10
        flg.stride = 1
        flg.exercises = ['ex1']
        flg.t_subjects = None

        tr, ts = data_generator.generate_datasets(flg)

        self.assertTrue(os.path.exists(
            os.path.join(self.gen_path, 'test.csv')))

        df = pd.read_csv(os.path.join(self.gen_path, 'test.csv'))

        self.assertEqual(61, df.shape[1])
        self.assertEqual(182, df.shape[0])


if __name__ == '__main__':
    unittest.main()
