#!/bin/bash
set -Eeuo pipefail

yum install -y gcc make rpm-build libtool hwloc-devel \
  libX11-devel libXt-devel libedit-devel libical-devel \
  ncurses-devel perl postgresql-devel postgresql-contrib python-devel tcl-devel \
  tk-devel swig expat-devel openssl-devel libXext libXft \
  autoconf automake

yum install -y expat libedit postgresql-server postgresql-contrib python \
  sendmail sudo tcl tk libical

git clone https://github.com/pbspro/pbspro.git /src/pbspro
cd /src/pbspro
./autogen.sh
./configure -prefix=/opt/pbs
make
make dist
# directories for rpm build must be ~/rpmbuild
mkdir /root/rpmbuild
mkdir /root/rpmbuild/SOURCES
mkdir /root/rpmbuild/SPECS
cp pbspro-*.tar.gz /root/rpmbuild/SOURCES
cp pbspro.spec /root/rpmbuild/SPECS
# build rpms
cd /root/rpmbuild/SPECS
rpmbuild -ba pbspro.spec
# install pbspro
cd /root/rpmbuild/RPMS/x86_64
yum install -y pbspro-server-*.rpm


# on startup
# /etc/init.d/pbs start
# . /etc/profile.d/pbs.sh

# qmgr -c "set server flatuid=true"
# qmgr -c "set server acl_roots+=root@*"
# qmgr -c "set server operators+=root@*"