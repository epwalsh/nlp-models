debug         = 0
test          = nlpete scripts
port          = 5000
COVERAGE     := $(addprefix --cov=, $(test))
DATADIR       = data
DATASETS     := $(wildcard $(DATADIR)/*.tar.gz)
EXPERIMENTDIR = experiments
EXPERIMENTS  := $(wildcard $(EXPERIMENTDIR)/**/*.json)
BENCHMARKDIR  = scripts/benchmarking
BENCHMARKS   := $(wildcard $(BENCHMARKDIR)/*.py)

#
# Training commands.
#

.PHONY : train
train :
ifeq ($(debug),0)
	@pipenv run ./scripts/training/train.sh
else
	@echo "<=== Debug mode ===>"
	@CUDA_LAUNCH_BLOCKING=1 pipenv run ./scripts/training/train.sh
endif

.PHONY : tensorboard
tensorboard :
	@TB_PORT=$(port) TB_LOGDIR=$(logdir) pipenv run ./scripts/training/tensorboard.sh

.PHONY : vocab
vocab :
	@pipenv run ./scripts/training/make_vocab.sh

# Need this to force targets to build, even when the target file exists.
.PHONY : phony-target

$(DATADIR)/%.tar.gz : phony-target
	@if ! [ -d $(patsubst %.tar.gz,%,$@) ]; then \
		echo "Extracting $@ to $(patsubst %.tar.gz,%,$@)"; \
		mkdir -p $(patsubst %.tar.gz,%,$@) && tar xzfv $@ -C $(patsubst %.tar.gz,%,$@) --strip-components 1; \
	fi

$(EXPERIMENTDIR)/%.json : phony-target
	pipenv run ./scripts/training/train.sh $@

#
# Testing commands.
#

.PHONY : typecheck
typecheck :
	@echo "Typechecks: mypy"
	@pipenv run mypy $(test) --ignore-missing-imports --no-site-packages

.PHONY : lint
lint :
	@echo "Lint: pydocstyle"
	@pipenv run pydocstyle --config=.pydocstyle $(test)
	@echo "Lint: flake8"
	@pipenv run flake8 $(test)
	@echo "Lint: black"
	@pipenv run black --check --diff $(test)

.PHONY : unit-test
unit-test :
	@echo "Unit tests: pytest"
ifneq ($(findstring test,$(test)),)
	@pipenv run python -m pytest -v --color=yes $(test)
else
	@pipenv run python -m pytest -v --cov-config .coveragerc $(COVERAGE) --color=yes $(test)
endif

.PHONY : check-scripts
check-scripts :
	pipenv run ./scripts/checks/run_all.sh \
		./scripts/checks/check_links.py \
		./scripts/checks/check_whitespace.sh \
		$(BENCHMARKS)

.PHONY : test
test : typecheck lint unit-test

.PHONY : format
format :
	black $(test)

#
# Git helpers.
#

.PHONY: create-branch
create-branch :
ifneq ($(issue),)
	git checkout -b ISSUE-$(issue)
	git push --set-upstream origin ISSUE-$(issue)
else ifneq ($(name),)
	git checkout -b $(name)
	git push --set-upstream origin $(name)
else
	$(error must supply 'issue' or 'name' parameter)
endif
