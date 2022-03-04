packages_by_priority := webknossos wkcuber cluster_tools docs
packages_by_dependency := cluster_tools webknossos wkcuber docs

define in_each_pkg_by_dependency
  for PKG in $(packages_by_dependency); do echo $$PKG; cd $$PKG; $1; cd ..; done
endef

.PHONY: list_packages_by_priority update update-internal install

list_packages_by_priority:
	@echo $(packages_by_priority)

update:
	$(call in_each_pkg_by_dependency, poetry update --no-dev)

update-internal:
	$(call in_each_pkg_by_dependency, poetry update $(packages_by_dependency))

install:
	$(call in_each_pkg_by_dependency, poetry install)
