"""Reporter for training runs

This class can be used to store statistics related to the trainign of a
classifier. The generated report can be dumped into a JSON file whenever
the training completes.

A run is a collection of parameters and iterations. The parameters can be added
individually or in bulk, with a dictionary. Anyway, a run should be initialized
first before iterations can be added. The report itself is represented by a
ordered dictionary, so that the final JSON file will be more readable. However,
the order of the keys should never be used to implement any kind of logic in
the reporter, nor in modules that use the Reporter class.
"""

import datetime
import time
import json
import collections

import humanize


class Reporter():

    def __init__(self, destination, autosave=False, autosave_count=0):
        """Create a reporter object

        The constructor will initialize all the reporter fields.

        Arguments:

        destination -- the final destination of the report file
        autosave -- if True, the report will be dumped automatically
        autosave_count -- the number of iterations after which the report
                          will be saved (defaulted to 0, no autosave)
        """
        self.destination = destination
        self.autosave = autosave
        self.autosave_count = autosave_count
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
        """Add results from an iteration

        This method saves in the report a set of results from an iteration. The
        reporter automatically adds the end time of the iteration, and the
        elapsed time, both in a timestamp format and in a natural fashion.

        Arguments:

        iteration -- a dictionary containing serializable keys
        """
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
        num_iterations = len(self.report['iterations'])

        if self.autosave and num_iterations % self.autosave_count == 0:
            print('autodumping...')
            self.dump_report()

    def dump_report(self):
        """Save the report in JSON format

        This method dumps the report in a JSON file. The ending time and the
        elapsed time are automatically included in the report.
        """
        end = datetime.datetime.now()
        start = self.report.get('start_time', None)
        elapsed, elapsed_natural = self.get_elapsed(start, end)

        self.report['end_time'] = end
        self.report['elapsed'] = elapsed
        self.report['elapsed_natural'] = elapsed_natural

        with open(self.destination, 'w') as r:
            json.dump(self.report, r, default=self.__convert_dates, indent=2)

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

    def __convert_dates(self, o):
        return o.__str__() if isinstance(o, datetime.datetime) else str(o)
