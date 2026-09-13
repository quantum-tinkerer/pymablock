# Integration artifact

`astro-myst-notebooks-0.2.0.tgz` is built from the sibling repository at
`~/src/astro-myst-notebooks` using `pixi run pack`. It contains compiled code,
declarations, and the Jupyter adapter. The site installs it through package.json
and package-lock.json; building this checkout needs no sibling directory.

Current artifact source: commit `ae4996c` in that repository.

After changing the integration, repack it and replace this artifact. Run
`npm install ./vendor/astro-myst-notebooks-0.2.0.tgz` from docs/starlight to update
the installation and lockfile integrity, then run the build and browser checks.
Use a new package version for subsequent released artifacts.
