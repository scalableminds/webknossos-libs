#!/bin/bash

set -eEuo pipefail

./format.sh
./lint.sh
./typecheck.sh
