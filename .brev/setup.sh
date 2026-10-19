#!/bin/bash
# Create a wrapper that installs deps before starting Jupyter
cat > /home/ubuntu/start.sh << 'EOF'
#!/bin/bash
cd /home/ubuntu/nim-workshop
curl -sS https://bootstrap.pypa.io/get-pip.py | /home/ubuntu/.venv/bin/python
/home/ubuntu/.venv/bin/pip install -r docker/requirements.txt
jupyter-lab --allow-root --NotebookApp.token='' --NotebookApp.base_url='/notebook/' --ip='0.0.0.0' --port=8888 --NotebookApp.disable_check_xsrf=True
EOF
chmod +x /home/ubuntu/start.sh
bash /home/ubuntu/start.sh
