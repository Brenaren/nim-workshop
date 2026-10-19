#!/bin/bash

# Create an IPython startup script that installs deps on first kernel launch
mkdir -p /home/ubuntu/.ipython/profile_default/startup
cat > /home/ubuntu/.ipython/profile_default/startup/00-install-deps.py << 'PYEOF'
import subprocess, sys, os

flag = os.path.expanduser('~/.nim_workshop_deps_installed')
if not os.path.exists(flag):
    # Bootstrap pip if missing
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

# Start Jupyter
cd /home/ubuntu/nim-workshop
jupyter-lab --allow-root --NotebookApp.token='' --NotebookApp.base_url='/notebook/' \
  --ip='0.0.0.0' --port=8888 --NotebookApp.disable_check_xsrf=True
