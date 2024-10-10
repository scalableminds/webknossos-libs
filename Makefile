packages_by_priority := webknossos cluster_tools docs
packages_by_dependency := cluster_tools webknossos docs
code_packages := cluster_tools webknossos

define in_each_pkg_by_dependency
  for PKG in $(packages_by_dependency); do echo $$PKG; cd $$PKG; $1; cd ..; done
endef

define in_each_code_pkg
  for PKG in $(code_packages); do echo $$PKG; cd $$PKG; $1; cd ..; done
endef

.PHONY: list_packages_by_priority update update-internal install format lint typecheck flt test

list_packages_by_priority:
	@echo $(packages_by_priority)

install:
	$(call in_each_pkg_by_dependency, uv sync --all-extras)

format:
	$(call in_each_code_pkg, ./format.sh)

lint:
	$(call in_each_code_pkg, ./lint.sh)

typecheck:
	$(call in_each_code_pkg, ./typecheck.sh)

flt:
	$(call in_each_code_pkg, ./format.sh && ./lint.sh && ./typecheck.sh)

test:
	$(call in_each_code_pkg, ./test.sh)
