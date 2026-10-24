# Checks no workflow runs as written: every optional feature, including the ones CI leaves pending
# because it builds them against an empty stub (.github/feature-check-pending.txt)
# @quick
python3 scripts/feature_check.py
