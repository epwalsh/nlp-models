debug         = 0
test          = nlpete scripts
port          = 5000
COVERAGE     := $(addprefix --cov=, $(test))
PYTHONPATH    = allennlp
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
	@./scripts/training/train.sh
else
	@echo "<=== Debug mode ===>"
	@CUDA_LAUNCH_BLOCKING=1 ./scripts/training/train.sh
endif

.PHONY : tensorboard
tensorboard :
	@TB_PORT=$(port) TB_LOGDIR=$(logdir) ./scripts/training/tensorboard.sh

.PHONY : vocab
vocab :
	@./scripts/training/make_vocab.sh

# Need this to force targets to build, even when the target file exists.
.PHONY : phony-target

$(DATADIR)/%.tar.gz : phony-target
	@if ! [ -d $(patsubst %.tar.gz,%,$@) ]; then \
		echo "Extracting $@ to $(patsubst %.tar.gz,%,$@)"; \
		mkdir -p $(patsubst %.tar.gz,%,$@) && tar xzfv $@ -C $(patsubst %.tar.gz,%,$@) --strip-components 1; \
	fi

$(EXPERIMENTDIR)/%.json : phony-target
	./scripts/training/train.sh $@

#
# Testing commands.
#

.PHONY : typecheck
typecheck :
	@echo "Typechecks: mypy"
	@PYTHONPATH=$(PYTHONPATH) mypy $(test) --ignore-missing-imports --no-site-packages

.PHONY : lint
lint :
	@echo "Lint: pydocstyle"
	@pydocstyle --config=.pydocstyle $(test)
	@echo "Lint: flake8"
	@PYTHONPATH=$(PYTHONPATH) flake8 $(test)
	@echo "Lint: black"
	@black --check $(test)

.PHONY : unit-test
unit-test :
	@echo "Unit tests: pytest"
ifneq ($(findstring test,$(test)),)
	PYTHONPATH=$(PYTHONPATH) python -m pytest -v --color=yes $(test)
else
	PYTHONPATH=$(PYTHONPATH) python -m pytest -v --cov-config .coveragerc $(COVERAGE) --color=yes $(test)
endif

.PHONY : check-scripts
check-scripts :
	./scripts/checks/run_all.sh \
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
