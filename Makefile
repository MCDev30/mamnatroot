PYTHON = $(shell poetry env info --path)/bin/python3

all: build

# Cible pour nettoyer le projet des fichiers générés
.PHONY: clean
clean:
	@rm -rf build/ dist/ src/mamnatroot.egg-info/
	@find . -name "*.so" -delete
	@find . -name "*.c" -path "*mamnatroot*" -delete
	

.PHONY: build
build: clean
	@$(PYTHON) setup.py build_ext --inplace
	@clear

.PHONY: run
run:
	@if [ -z "$(file)" ]; then \
		echo "Erreur : Nom de fichier manquant. Utilisation : make run file=<nom_du_fichier_sans_extension>"; \
		exit 1; \
	fi
	@echo "--- Exécution de $(file).py ---"
	@$(PYTHON) $(file).py

.PHONY: dist
dist: build
	@echo "--- Construction des packages de distribution (sdist et wheel) ---"
	@$(PYTHON) -m build
	@echo "Packages créés dans le dossier 'dist/'."

# Cible pour déployer sur TestPyPI
# Dépend de 'dist' pour s'assurer que les packages sont construits avant.
.PHONY: deploy-test
deploy-test: dist
	@echo "--- Déploiement sur TestPyPI ---"
	@$(PYTHON) -m twine upload --repository testpypi dist/*

# Cible pour déployer sur le vrai PyPI
.PHONY: deploy
deploy: dist
	@echo "--- DÉPLOIEMENT SUR PYPI (PRODUCTION) ---"
	@$(PYTHON) -m twine upload dist/*
