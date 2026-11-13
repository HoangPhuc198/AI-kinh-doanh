"""Hồi quy cho lỗi "torch can only be initialized once per process"."""
import importlib.util
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from tests._paths import PROJECT, APP

FAKE_TORCH = '''
import sys
if getattr(sys, "_torch_import_started", False):
    raise ImportError("`torch` can only be initialized once per process")
sys._torch_import_started = True
sys._fake_torch_runs = getattr(sys, "_fake_torch_runs", 0) + 1
raise OSError("WinError 126: dll gia lap thieu")
'''


class TorchLifecycle(unittest.TestCase):
    def test_failed_import_is_never_retried_and_keeps_root_cause(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / 'torch'
            pkg.mkdir()
            (pkg / '__init__.py').write_text(FAKE_TORCH)
            script = textwrap.dedent(f'''
                import sys
                sys.path[:0] = [{tmp!r}, {str(APP)!r}]
                import runtime
                errs = []
                for _ in range(3):
                    try:
                        runtime.load_torch()
                    except BaseException as e:
                        errs.append(e)
                assert isinstance(errs[0], OSError), errs[0]
                for e in errs[1:]:
                    assert isinstance(e, runtime.TorchUnavailable), e
                    assert isinstance(e.__cause__, OSError) and 'WinError 126' in str(e), e
                    assert 'initialized once' not in str(e)
                assert sys._fake_torch_runs == 1, sys._fake_torch_runs
                print('OK')
            ''')
            r = subprocess.run([sys.executable, '-I', '-c', script], capture_output=True, text=True, cwd=tmp)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn('OK', r.stdout)

    def test_source_has_no_reload_or_sysmodules_hacks(self):
        bad = re.compile(r'importlib\.reload|reload\(\s*torch|sys\.modules\s*(\[|\.pop|\.__delitem__)|del\s+sys\.modules')
        for f in list(APP.rglob('*.py')) + list(APP.rglob('*.html')):
            self.assertIsNone(bad.search(f.read_text(encoding='utf-8')), f)

    def test_torch_imported_only_in_runtime_module(self):
        for f in APP.rglob('*.py'):
            if f.name == 'runtime.py':
                continue
            self.assertNotRegex(f.read_text(encoding='utf-8'),
                                r'(?m)^\s*(import torch|from torch|from ultralytics)', f)

    def test_real_torch_imports_in_clean_process(self):
        if importlib.util.find_spec('torch') is None:
            self.skipTest('torch chua cai')
        script = (f'import sys; sys.path.insert(0, {str(APP)!r}); import runtime; '
                  'runtime.load_torch(); runtime.load_torch(); print(runtime.torch_status()["ready"])')
        r = subprocess.run([sys.executable, '-I', '-c', script], capture_output=True, text=True, cwd=str(PROJECT))
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        self.assertIn('True', r.stdout)


if __name__ == '__main__':
    unittest.main()
