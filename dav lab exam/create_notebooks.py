import nbformat as nbf
import os

scripts = ['act1.py', 'act2.py', 'act3.py']

for py_file in scripts:
    with open(py_file, 'r') as f:
        code = f.read()
    
    nb = nbf.v4.new_notebook()
    nb['cells'] = [nbf.v4.new_code_cell(code)]
    
    ipynb_file = py_file.replace('.py', '.ipynb')
    with open(ipynb_file, 'w') as f:
        nbf.write(nb, f)
        
    print(f"Created {ipynb_file}")
