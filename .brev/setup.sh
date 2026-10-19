#!/bin/bash

# Create IPython startup script (runs when any kernel starts, after venv exists)
mkdir -p /home/ubuntu/.ipython/profile_default/startup
cat > /home/ubuntu/.ipython/profile_default/startup/00-install-deps.py << 'PYEOF'
import subprocess, sys, os

flag = os.path.expanduser('~/.nim_workshop_deps_installed')
if not os.path.exists(flag):
    try:
        import pip
    except ImportError:
        import urllib.request
        urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '/tmp/get-pip.py')
        subprocess.check_call([sys.executable, '/tmp/get-pip.py'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                           '-r', '/home/ubuntu/nim-workshop/docker/requirements.txt'])
    open(flag, 'w').close()
PYEOF

# Start Jupyter (venv already exists by this point)
cd /home/ubuntu/nim-workshop
jupyter-lab --allow-root --NotebookApp.token='' --NotebookApp.base_url='/notebook/' \
  --ip='0.0.0.0' --port=8888 --NotebookApp.disable_check_xsrf=True
