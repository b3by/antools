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
import os

import humanize


class Reporter():
    """Reporter class for automated report generation

    This class provides facilities for logging and storing results of iterative
    executions. A reporter can be set to automatically dump the results after
    a specific number of executions, and can easily restore a previously
    started report.

    Attributes
    ----------
    report : collections.OrderedDict
        This attribute contains the report itself. In order to preserve the
        inclusion order of the items in the report, an ordered dictionary is
        used.
    destination : str
        The destination of the report to dump. It should specify a full path
        with the inclusion of the file name.
    autosave : bool
        A flag that triggers the autosave after a specified number of
        iterations. When set to true, the autosave_count should be not zero.
    autosave_count : int
        The number of iterations to wait for the autosave.
    """

    def __init__(self, destination, restore=True, autosave=False,
                 autosave_count=0):
        """Create a reporter object

        The constructor will initialize all the reporter fields.

        Parameters
        ----------
        destination : str
            The final destination of the report file.
        restore : bool
            If True, an existing report at the given destination will be fully
            restored.
        autosave : bool
            If True, the report will be dumped automatically.
        autosave_count : int
            The number of iterations after which the report will be saved
            automatically. By default, it is set to 0, so no autosave is
            enabled.
        """
        self.destination = destination
        self.autosave = autosave
        self.autosave_count = autosave_count

        if restore and os.path.isfile(self.destination):
            with open(self.destination, 'r') as f:
                self.report = collections.OrderedDict(
                    json.loads(f.read()))
                self.report['start_time'] = datetime.datetime.strptime(
                    self.report['start_time'], '%Y-%m-%d %H:%M:%S.%f')

                if self.report.get('iterations', None) is not None:
                    for it in self.report['iterations']:
                        it['end_time'] = datetime.datetime.strptime(
                            self.report['end_time'], '%Y-%m-%d %H:%M:%S.%f')

                self.__run = True
        else:
            self.report = collections.OrderedDict()
            self.__run = False

    def spawn_run(self, run_name):
        """Spawn a run within the report

        This method initializes a run within the report. The run name is
        required.

        Parameters
        ---------
        run_name : str
            The name of the run to create.
        """
        self.report['run_name'] = run_name
        self.report['start_time'] = datetime.datetime.now()
        self.report['params'] = {}
        self.__run = True

    def add_run_param(self, key, value):
        """Add new run parameter

        This method creates a new parameter for the run. The parameter will be
        included in the 'params' dictionary within the report.

        Parameters
        ----------
        key
            The key, that is, the name of the new parameter.
        value
            The value, that is, the value of the new parameter.
        """
        self.__validate_run()
        self.report['params'][key] = value

    def add_run_params(self, params):
        """Add set of run parameters

        This method adds run parameters in bulk, by taking them from the
        provided dictionary.

        Parameters
        ----------
        params : dict
            The dictionary to copy within the params dictionary.
        """
        for param in params:
            self.add_run_param(param, params[param])

    def add_iteration(self, iteration):
        """Add results from an iteration

        This method saves in the report a set of results from an iteration. The
        reporter automatically adds the end time of the iteration, and the
        elapsed time, both in a timestamp format and in a natural fashion.

        Parameters
        ----------
        iteration : dict
            A dictionary containing serializable keys.
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
        """Compute elapsed time between two datetime objects

        This method receives two datetime objects and computes the delta
        between them. It then returns both the delta and the humanized delta,
        that is, a string representing the time delta in natural form.

        Parameters
        ----------
        start : datetime.datetime
            The start time of the event that should be measured.
        end : datetime.datetime
            The end time of the event that should be measured.

        Returns
        -------
        tuple
            A tuple containing the elapsed time delta as first element and the
            humanized time delta as second element.
        """
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
