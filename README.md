# antools

[![forthebadge](https://forthebadge.com/images/badges/no-ragrets.svg)](https://forthebadge.com)
[![Travis](https://img.shields.io/travis/b3by/antools.svg?style=for-the-badge)](https://travis-ci.org/b3by/antools)

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

### `net_utils`
This module contains utilities for the implementation of models. The APIs are
quite simple, and do not involve classes of any sort.

The available APIs include:

- [`dense_layer(x, number_units, name, activation='relu')`](https://github.com/b3by/antools/blob/0aa864f8c2b5b8fe0db66e28f2237914ac8a7014/antools/network/net_utils.py#L34)
- [`drop_layer(x, keep_probability, name)`](https://github.com/b3by/antools/blob/0aa864f8c2b5b8fe0db66e28f2237914ac8a7014/antools/network/net_utils.py#L59)
- [`dense_layers(x, units, keep_probability, name_prefix, activation='relu')`](https://github.com/b3by/antools/blob/0aa864f8c2b5b8fe0db66e28f2237914ac8a7014/antools/network/net_utils.py#L77)
- [`depth_conv2d_layer(x, kernel, name, padding='SAME')`](https://github.com/b3by/antools/blob/0aa864f8c2b5b8fe0db66e28f2237914ac8a7014/antools/network/net_utils.py#L106)
- [`maxpool_layer(x, kernel, name, padding='VALID')`](https://github.com/b3by/antools/blob/0aa864f8c2b5b8fe0db66e28f2237914ac8a7014/antools/network/net_utils.py#L138)
- [`softmax_layer(x, number_labels, name)`](https://github.com/b3by/antools/blob/0aa864f8c2b5b8fe0db66e28f2237914ac8a7014/antools/network/net_utils.py#L158)
- [`cross_entropy_loss(logits, labels, name, regularize=False)`](https://github.com/b3by/antools/blob/0aa864f8c2b5b8fe0db66e28f2237914ac8a7014/antools/network/net_utils.py#L178)
- [`soft_cross_entropy_loss(last_layer, labels, name)`](https://github.com/b3by/antools/blob/0aa864f8c2b5b8fe0db66e28f2237914ac8a7014/antools/network/net_utils.py#L203)
- [`adam_backprop(loss, learning_rate, global_step, name)`](https://github.com/b3by/antools/blob/0aa864f8c2b5b8fe0db66e28f2237914ac8a7014/antools/network/net_utils.py#L223)
- [`batchnorm_layer(x, n_out, is_train, name)`](https://github.com/b3by/antools/blob/0aa864f8c2b5b8fe0db66e28f2237914ac8a7014/antools/network/net_utils.py#L249)

The high level functions in this module allow to create layers of different
kinds. For instance, the following code generates a sandwich of three
convolutional layers, interleaved with max pooling layers. Batchnorm is also
included in the stack.

```python
from types import SimpleNamespace

from antools.network import net_utils

kernels = [
    SimpleNamespace(kernel=[4, 4], strides=[1, 1, 1, 1], depth=10),
    SimpleNamespace(kernel=[3, 3], strides=[1, 1, 1, 1], depth=2),
    SimpleNamespace(kernel=[2, 2], strides=[1, 1, 1, 1], depth=1)
]

pools = [
    SimpleNamespace(kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1]),
    SimpleNamespace(kernel=[1, 2, 2, 1], strides=[1, 1, 1, 1]),
    SimpleNamespace(kernel=[1, 2, 2, 1], strides=[1, 1, 1, 1])
]

# X -- input tensor
layers = [X]

for idx, (k, p) in enumerate(zip(kernels, pools)):
    ch = layers[-1].shape[3].value

    c_name = 'conv{}'.format(idx + 1)
    n_name = 'norm{}'.format(idx + 1)
    p_name = 'pool{}'.format(idx + 1)

    c = net_utils.depth_conv2d_layer(layers[-1], k, c_name)
    n = net_utils.batchnorm_layer(c, ch * k.depth, is_train, n_name)
    m = net_utils.maxpool_layer(n, p, p_name)

    layers.append(c)
    layers.append(n)
    layers.append(m)

# at this points, all the layers are saved in the layers list
```

Another high level API is `dense_layers`, which allows to generate fully
connected stacks by providing the number of units and the dropout probabiliy.

```python
# x -- input tensor
# keep_prob -- tensor variable for the dropout

units = (500, 250, 125)
dense_output = net_utils.dense_layers(X, units, keep_prob, 'dense',
                                      activation='tanh')
```

### `clprint`
This module is quite useless. It lets you print colored stuff on the terminal.
Here follows an example script.

```python
from antools.utils import clprint

print(clprint.bold('This text is bold... it has guts!'))
print(clprint.ok('This text is green... such envy!'))
print(clprint.info('This text is blue... is it Monday already?'))
print(clprint.warn('This text is orange... because oranges.'))
print(clprint.err('This text is red... the batch better have mah money!'))
```

![c1](https://raw.githubusercontent.com/b3by/antools/master/images/color1.png)

Alternatively, messages can be decorated with a prefix:

```python
from antools.utils import clprint

clprint.ptinfo('This is an info during a run...')
clprint.ptrun('Something hairy happened?')
clprint.ptok('Oh no, my bad, everything is alright.')
```

![c2](https://raw.githubusercontent.com/b3by/antools/master/images/color2.png)

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
