# Install segtools
pip3 install -r requirements.txt
pip3 install tensorflow
source ./set_pythonpath.sh

# Upgrade pip
pip3 install --upgrade pip


# Update python to 3.6 and make it default (execute on all nodes!!)
yum install -y centos-release-scl
yum install -y rh-python36
scl enable rh-python36 bash
rm -f /usr/bin/python3
ln -s $(which python3) /usr/bin/python3

# Setup clusterfutures (execute on all nodes!!)
cd /clusterfutures
python3 setup.py install

# Execute
python3 tools/distribute.py with tool=predict strategy=slurm config=configs/prediction/base.yaml                                                             



# special sbatch lines for local docker
"pip3 install --upgrade pip",
"pip3 uninstall futures",
"pip3 install futures",
"python3 setup.py install",
"pushd /segtools",
"echo $(pwd)",
"pip3 install -r requirements.txt",
"pip3 install tensorflow",
"source ./set_pythonpath.sh",
"popd",