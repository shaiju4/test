
from nbconvert import PythonExporter
import nbformat

def run_ipynb(file_path):
    with open(file_path) as f:
        nb = nbformat.read(f, as_version=4)
    exporter = PythonExporter()
    source, _ = exporter.from_notebook_node(nb)
    exec(source)

# Example usage
ipynb_file_path = "test.ipynb"
run_ipynb(ipynb_file_path)

