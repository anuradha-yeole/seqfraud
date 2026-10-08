.PHONY: data train bench serve test all

data:
	python -m seqfraud.simulate --out data/transactions.parquet

train:
	python -m seqfraud.train --data data/transactions.parquet --out artifacts

bench:
	python -m seqfraud.benchmark --data data/transactions.parquet --artifacts artifacts

serve:
	uvicorn seqfraud.serve:app --host 0.0.0.0 --port 8000

test:
	pytest -q

all: data train bench
