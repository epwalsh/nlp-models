test          = modules
COVERAGE     := $(addprefix --cov=, $(test))
PYTHONPATH    = allennlp
DATADIR       = data
DATASETS     := $(wildcard $(DATADIR)/*.tar.gz)
EXPERIMENTDIR = experiments
EXPERIMENTS  := $(wildcard $(EXPERIMENTDIR)/**/*.json)
BAR := $(dir $(EXPERIMENTS))

#
# Training commands.
#

.PHONY : train
train :
	./scripts/train.sh

# Need this to force targets to build, even when the target file exists.
.PHONY : phony-target

$(DATADIR)/%.tar.gz : phony-target
	@if ! [ -d $(patsubst %.tar.gz,%,$@) ]; then \
		echo "Extracting $@ to $(patsubst %.tar.gz,%,$@)"; \
		tar xzf $@ -C $(DATADIR); \
	fi

# Experiments depend on their datasets.
.SECONDEXPANSION:
$(EXPERIMENTDIR)/%.json : data/$$(shell dirname %.json).tar.gz
	./scripts/train.sh $@

#
# Testing commands.
#

.PHONY : typecheck
typecheck :
	@echo "Typechecks: mypy"
	@PYTHONPATH=$(PYTHONPATH) mypy $(test) --ignore-missing-imports

.PHONY : lint
lint :
	@echo "Lint: pydocstyle"
	@pydocstyle --config=.pydocstyle $(test)
	@echo "Lint: pylint"
	@PYTHONPATH=$(PYTHONPATH) pylint --rcfile=.pylintrc -f colorized $(test)

.PHONY : unit-test
unit-test :
	@echo "Unit tests: pytest"
ifeq ($(suffix $(test)),.py)
	PYTHONPATH=$(PYTHONPATH) python -m pytest -v --color=yes $(test)
else
	PYTHONPATH=$(PYTHONPATH) python -m pytest -v --cov-config .coveragerc $(COVERAGE) --color=yes $(test)
endif

.PHONY : test
test : typecheck lint unit-test

#
# Git helpers.
#

.PHONY: create-branch
create-branch :
ifeq ($(issue),)
	$(error must supply 'issue' parameter)
else
	git checkout -b ISSUE-$(issue)
	git push --set-upstream origin ISSUE-$(issue)
endif
