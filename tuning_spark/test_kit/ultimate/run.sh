#/bin/bash
stop-all.sh
python3 run-bo.py
stop-all.sh
python3 run-bo-pp-replace-help.py
stop-all.sh
python3 run-bo-bliss.py
stop-all.sh
python3 run-restune.py
