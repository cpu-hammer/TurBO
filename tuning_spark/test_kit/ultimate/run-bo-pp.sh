#!/bin/bash
stop-all.sh
python3 run-bo-allpp.py 0.05
stop-all.sh
python3 run-bo-allpp.py 0.1
stop-all.sh
python3 run-bo-allpp.py 0.01
stop-all.sh
python3 run-bo-halfpp.py 0.05
stop-all.sh
python3 run-bo-halfpp.py 0.1
stop-all.sh
python3 run-bo-halfpp.py 0.01
