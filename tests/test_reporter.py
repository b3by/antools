import unittest

from antools.utils import reporter


class TestReporter(unittest.TestCase):

    def test_create_reporter(self):
        r = reporter.Reporter('./wololo.json')
        self.assertEqual(r.destination, './wololo.json')
        self.assertEqual(r.report, {})

    def test_spawn_run(self):
        r = reporter.Reporter('./wololo.json')
        r.spawn_run('Test run')
        self.assertEqual(r.report['run_name'], 'Test run')
        self.assertIsNotNone(r.report['start_time'])
        self.assertEqual(r.report['params'], {})

    def test_add_run_param(self):
        r = reporter.Reporter('./wololo.json')
        r.spawn_run('Test run')
        r.add_run_param('test_param', 0.01)
        r.add_run_param('test_param_2', [10, 20, 30])
        self.assertEqual(r.report['params']['test_param'], 0.01)
        self.assertEqual(r.report['params']['test_param_2'], [10, 20, 30])

    def test_add_run_param_no_run(self):
        r = reporter.Reporter('./wololo.json')

        with self.assertRaises(Exception) as context:
            r.add_run_param('test_param', 0.001)

        self.assertTrue('No run initialized yet.' in str(context.exception))

    def test_add_run_params(self):
        r = reporter.Reporter('./wololo.json')
        r.spawn_run('Test run')
        r.add_run_params({'param_1': 10, 'param_2': {'test_k': 'v'}})
        self.assertEqual(r.report['params']['param_1'], 10)
        self.assertEqual(r.report['params']['param_2'], {'test_k': 'v'})

    def test_add_run_params_no_run(self):
        r = reporter.Reporter('./wololo.json')

        with self.assertRaises(Exception) as context:
            r.add_run_params({'param_1': 10, 'param_2': {'test_k': 'v'}})

        self.assertTrue('No run initialized yet.' in str(context.exception))

    def test_dict(self):
        r = reporter.Reporter('./wololo.json')
        r.spawn_run('Test run')
        r.add_run_param('test_p', 100)
        test_d = dict(r)
        self.assertEqual(test_d['run_name'], 'Test run')
        self.assertEqual(test_d['params'], {'test_p': 100})


if __name__ == '__main__':
    unittest.main()
