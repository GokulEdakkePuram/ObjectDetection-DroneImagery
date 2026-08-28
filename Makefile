.PHONY: help setup data profile smoke baseline finetune eval test lint clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## install dependencies into .venv
	uv sync --extra dev

data:  ## download and convert VisDrone (~2 GB)
	uv run aerialdet download

profile:  ## measure class balance and object scale
	uv run aerialdet profile

smoke:  ## 1-epoch run to prove the pipeline works
	uv run aerialdet train smoke

baseline:  ## control run at 640 px
	uv run aerialdet train baseline_640

finetune:  ## main run at 960 px
	uv run aerialdet train finetune_960

eval:  ## compare both runs on the val split
	uv run aerialdet eval \
		runs/train/baseline_640/weights/best.pt \
		runs/train/finetune_960/weights/best.pt

test:  ## run the test suite
	uv run pytest

lint:  ## lint and format check
	uv run ruff check src tests
	uv run ruff format --check src tests

clean:  ## remove generated runs and reports
	rm -rf runs reports
