.ONESHELL:
.PHONY: force-build


notes: force-build
	cd notes
	pdflatex notes
	bibtex notes
	pdflatex notes
	pdflatex notes


clean:
	find . -name "*.aux" -type f -delete
	find . -name "*.bbl" -type f -delete
	find . -name "*.blg" -type f -delete
	find . -name "*.log" -type f -delete
	find . -name "*.out" -type f -delete
	find . -name "*.toc" -type f -delete
