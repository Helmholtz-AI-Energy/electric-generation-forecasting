"""Run test script and inspect results."""

import argparse
import os
import logging

from notebook.notebookapp import NotebookApp

LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Run Jupyter notebook.')
parser.add_argument('--list', '-l', action='store_true',
                    help='List notebooks')

parser.add_argument('name', metavar='NAME', type=str, nargs='?',
                    help='Name of the notebook to run')

args = parser.parse_args()

# if not args.list and args.name is None:
#     parser.error("NAME is required")

if args.list:
    for file in sorted(os.listdir(os.path.dirname(__file__))):
        basename, extension = os.path.splitext(file)

        if not extension == '.ipynb':
            continue

        if basename in ['empty', 'template']:
            continue

        print(basename)

    parser.exit(status=0)

if args.name is not None:
    notebook_file = f".\\notebooks\\{args.name}.ipynb"
    if not os.path.isfile(notebook_file):
        import shutil
        shutil.copyfile('.\\notebooks\\empty.ipynb', notebook_file)

    argv = [notebook_file]
else:
    argv = []

os.environ['PROJECT_HOME'] = os.getcwd()
os.environ['NOTEBOOK_HOME'] = os.path.abspath('./notebooks')

app = NotebookApp()
app.launch_instance(argv)
