"""Reporter for training runs

This class can be used to store statistics related to the trainign of a
classifier. The generated report can be dumped into a JSON file whenever
the training complete.

A run is a collection of parameters and iterations. The parameters can be added
individually or in bulk, with a dictionary. Anyway, a run should be initialized
first.
"""

import datetime
import time
import json
import collections

import humanize


class Reporter():

    def __init__(self, destination):
        self.destination = destination
        self.report = collections.OrderedDict()
        self.__run = False

    def spawn_run(self, run_name):
        """Spawn a run within the report

        This method initializes a run within the report. The run name is
        required.

        Arguments:

        run_name -- the name of the run to create
        """
        self.report['run_name'] = run_name
        self.report['start_time'] = datetime.datetime.now()
        self.report['params'] = {}
        self.__run = True

    def add_run_param(self, k, v):
        """Add new run parameter

        This method creates a new parameter for the run. The parameter will be
        included in the 'params' dictionary within the report.

        Arguments:

        k -- the key, that is, the name of the new parameter
        v -- the value, that is, the value of the new parameter
        """
        self.__validate_run()
        self.report['params'][k] = v

    def add_run_params(self, d):
        """Add set of run parameters

        This method adds run parameters in bulk, by taking them from the
        provided dictionary.

        Arguments:

        d -- the dictionary to copy within the params dictionary
        """
        for param in d:
            self.add_run_param(param, d[param])

    def add_iteration(self, iteration):
        self.__validate_run()

        if self.report.get('iterations') is None:
            self.report['iterations'] = []

        end = datetime.datetime.now()
        iteration['end_time'] = end

        if len(self.report['iterations']) > 0:
            start = self.report['iterations'][-1]['end_time']
        else:
            start = self.report['start_time']

        elapsed, elapsed_natural = self.get_elapsed(start, end)
        iteration['elapsed'] = elapsed
        iteration['elapsed_natural'] = elapsed_natural

        self.report['iterations'].append(iteration)

    def dump_report(self):
        end = datetime.datetime.now()
        start = self.report['start_time']
        elapsed, elapsed_natural = self.get_elapsed(start, end)

        self.report['elapsed'] = elapsed
        self.report['elapsed_natural'] = elapsed_natural

        with open(self.destination, 'w') as r:
            json.dump(self.report, r, default=self.__convert_dates__, indent=2)

    def get_elapsed(self, start, end):
        elapsed = end - start
        elapsed_natural = humanize.naturaldelta(
            time.mktime(end.timetuple()) - time.mktime(start.timetuple()))

        return elapsed, elapsed_natural

    def __iter__(self):
        for k in self.report:
            yield k, self.report[k]

    def __validate_run(self):
        if self.report.get('params', None) is None:
            raise Exception('No run initialized yet.')

    def __convert_dates__(self, o):
        if isinstance(o, datetime.datetime):
            return o.__str__()
        else:
            try:
                return float(o)
            except Exception:
                try:
                    return int(o)
                except Exception:
                    return str(o)
