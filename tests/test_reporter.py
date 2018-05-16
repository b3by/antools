import unittest

import datetime
import time
import json
import os

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
        empty_d = dict(r)

        self.assertEqual({}, empty_d)

        r.spawn_run('Test run')
        r.add_run_param('test_p', 100)
        test_d = dict(r)

        self.assertEqual(test_d['run_name'], 'Test run')
        self.assertEqual(test_d['params'], {'test_p': 100})

    def test_add_iteration(self):
        r = reporter.Reporter('./wololo.json')

        r.spawn_run('Test run')
        r.add_iteration({'yolo': True})
        test_d = dict(r)

        self.assertEqual(1, len(test_d['iterations']))

        iteration = test_d['iterations'][0]

        self.assertEqual(True, iteration['yolo'])
        self.assertIsInstance(iteration['end_time'], datetime.datetime)
        self.assertIsInstance(iteration['elapsed'], datetime.timedelta)
        self.assertIsNotNone(iteration.get('elapsed_natural', None))

    def test_add_multiple_iterations(self):
        r = reporter.Reporter('./wololo.json')

        r.spawn_run('Test run')
        r.add_iteration({'first': True})
        time.sleep(5)
        r.add_iteration({'second': 10, 'value': 0.5})

        iterations = dict(r)['iterations']

        self.assertEqual(2, len(iterations))
        self.assertEqual(iterations[0]['first'], True)
        self.assertEqual(iterations[1]['second'], 10)
        self.assertEqual(iterations[1]['value'], 0.5)
        self.assertEqual(iterations[1]['elapsed_natural'], '5 seconds')

    def test_dump_report(self):
        r = reporter.Reporter('./tests/wololo.json')

        r.spawn_run('Test run')
        r.add_run_param('some_value', 0.5)
        r.add_iteration({'value': 100})
        r.dump_report()

        with open('./tests/wololo.json') as f:
            test_r = json.load(f)

        self.assertIsInstance(test_r, dict)
        self.assertEqual(test_r['run_name'], 'Test run')
        self.assertEqual(test_r['params']['some_value'], 0.5)
        self.assertEqual(len(test_r['iterations']), 1)
        self.assertIsNotNone(test_r['start_time'])
        self.assertIsNotNone(test_r['end_time'])
        self.assertIsNotNone(test_r['elapsed'])
        self.assertIsNotNone(test_r['elapsed_natural'])

        os.remove('./tests/wololo.json')

    def test_convert_dates(self):
        r = reporter.Reporter('./tests/wololo.json')

        r.spawn_run('Test run')
        r.add_iteration({'value': datetime.datetime.now()})
        r.add_iteration({'value': 10})
        r.add_iteration({'value': 4.2})
        r.dump_report()

        with open('./tests/wololo.json') as f:
            test_r = json.load(f)

        iterations = test_r['iterations']
        self.assertIsInstance(iterations[0]['value'], str)
        self.assertIsInstance(iterations[1]['value'], int)
        self.assertIsInstance(iterations[2]['value'], float)

        os.remove('./tests/wololo.json')

    def test_dump_interval(self):
        r = reporter.Reporter('./tests/wololo.json', autosave=True,
                              autosave_count=2)
        r.spawn_run('Test run')

        self.assertEqual(r.autosave_count, 2)
        self.assertFalse(os.path.exists('./tests/wololo.json'))
        self.assertFalse(os.path.isfile('./tests/wololo.json'))

        r.add_iteration({'value': 10})

        self.assertFalse(os.path.exists('./tests/wololo.json'))
        self.assertFalse(os.path.isfile('./tests/wololo.json'))

        r.add_iteration({'value': 10})

        self.assertTrue(os.path.exists('./tests/wololo.json'))
        self.assertTrue(os.path.isfile('./tests/wololo.json'))

        os.remove('./tests/wololo.json')


if __name__ == '__main__':
    unittest.main()
