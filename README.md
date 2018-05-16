# antools

[![forthebadge](https://forthebadge.com/images/badges/no-ragrets.svg)](https://forthebadge.com)
![Travis](https://img.shields.io/travis/b3by/antools.svg?style=for-the-badge)

A bit of boilerplating for data science projects.

## Content
There are different modules that you can import from the package, all at top
level:

```python
from antools.utils import data_generator
from antools.utils import reporter
from antools.utils import clprint
from antools.network import net_utils
```

### `reporter`
You can use the reporter to log and save the results of any complex iteration
of which you want to keep track. All the results are then saved into a JSON
file, for further analysis.

To create a reporter, you need to specify the destination of the file:

```python
rpt = reporter.Reporter('path/to/report.json')
```

If you want the reporter to save automatically every, say 5 iterations, you
can create a reporter like so:

```python
rpt = reporter.Reporter('path/to/report.json', autosave=True, autosave_count=5)
```

Once the reporter is created, you can spawn a run, store information related to
the execution at hand, and start filling the report with the iteration results.

```python
rpt.spawn_run('my_awesome_training')

# parameters can be added one by one
rpt.add_run_param('learning_rate', 0.00001)

# also dictionaries work for the parameters
other_params = {'epochs': 1000,'dropout_keep_prob': 0.5}
rpt.add_run_params(other_params)

for i in range(3):
    ...
    current_results = {'step': i, 'accuracy': i / 10, 'loss': i / 100}
    rpt.add_iteration(current_results)

# when we're done, we can dump the final report
rpt.dump_report()
```

The result of the previous code will look like this:

```json
{
  "run_name": "my_awesome_training",
  "start_time": "2018-05-16 20:59:48.674433",
  "params": {
    "learning_rate": 1e-05,
    "epochs": 1000,
    "dropout_keep_prob": 0.5
  },
  "iterations": [
    {
      "step": 0,
      "accuracy": 0.0,
      "loss": 0.0,
      "end_time": "2018-05-16 21:00:17.447882",
      "elapsed": "0:00:28.773449",
      "elapsed_natural": "29 seconds"
    },
    {
      "step": 1,
      "accuracy": 0.1,
      "loss": 0.01,
      "end_time": "2018-05-16 21:00:17.447982",
      "elapsed": "0:00:00.000100",
      "elapsed_natural": "a moment"
    },
    {
      "step": 2,
      "accuracy": 0.2,
      "loss": 0.02,
      "end_time": "2018-05-16 21:00:17.448010",
      "elapsed": "0:00:00.000028",
      "elapsed_natural": "a moment"
    }
  ],
  "end_time": "2018-05-16 21:00:21.593963",
  "elapsed": "0:00:32.919530",
  "elapsed_natural": "33 seconds"
}
```

## Installation
To install, use `pip` (a virtual environment is highly recommended):

```
pip install git+git://https://github.com/b3by/antools
```

If you use `pipenv`, you can install the repository like so:

```
pipenv install git+git://https://github.com/b3by/antools.git#egg=antools
```

Alternatively, you can clone the repository in your local machine, then install
it locally with either `pip` or `pipenv`:

```
pip install path/to/antools/
pipenv install path/to/antools/
```

## Testing
To run the tests, just type:

```bash
coverage run -m unittest discover -s tests/
```

If you want to check the coverage of a particular module, after the tests have
run, just type:

```bash
coverage report -m ./antools/utils/reporter.py
```
