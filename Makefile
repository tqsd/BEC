.PHONY: lint report handover

lint:
	lint-imports --config importlinter.ini

handover:
	python tools/handover.py --out HANDOVER.md

report: lint handover
