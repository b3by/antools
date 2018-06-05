import sys
import io
import unittest

from contextlib import contextmanager

from antools.utils import clprint


class TestReporter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_message = 'hi there'

    @contextmanager
    def capture_output(self):
        new_out, new_err = io.StringIO(), io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr

        try:
            sys.stdout, sys.stderr = new_out, new_err
            yield sys.stdout, sys.stderr
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    def test_bold_message(self):
        bold_message = clprint.bold(self.test_message)
        self.assertEqual(bold_message, '\033[1mhi there\033[0m')

    def test_info_message(self):
        info_message = clprint.info(self.test_message)
        self.assertEqual(info_message, '\033[94mhi there\033[0m')

    def test_warn_message(self):
        warn_message = clprint.warn(self.test_message)
        self.assertEqual(warn_message, '\033[93mhi there\033[0m')

    def test_err_message(self):
        err_message = clprint.err(self.test_message)
        self.assertEqual(err_message, '\033[91mhi there\033[0m')

    def test_ok_message(self):
        ok_message = clprint.ok(self.test_message)
        self.assertEqual(ok_message, '\033[92mhi there\033[0m')

    def test_mixed_colors(self):
        ok_bold_message  = clprint.ok(clprint.bold(self.test_message))
        self.assertEqual(ok_bold_message,
                         '\033[92m\033[1mhi there\033[0m\033[0m')

    def test_ptinfo(self):
        with self.capture_output() as (out, err):
            clprint.ptinfo(self.test_message)

        printed = out.getvalue().strip()
        self.assertEqual(printed, '\033[1m[~]\033[0m hi there')

    def test_ptrun(self):
        with self.capture_output() as (out, err):
            clprint.ptrun(self.test_message)

        printed = out.getvalue().strip()
        self.assertEqual(printed, '\033[1m\033[93m[~]\033[0m\033[0m hi there')

    def test_ptok(self):
        with self.capture_output() as (out, err):
            clprint.ptok(self.test_message)

        printed = out.getvalue().strip()
        self.assertEqual(printed, '\033[1m\033[92m[+]\033[0m\033[0m hi there')


if __name__ == '__main__':
    unittest.main()
