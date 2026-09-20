// Project choices live here; the integration contains no Pymablock paths/packages.
/** @type {import('starlight-pydocs').PydocsPackageInput} */
export const api = {
  name: 'pymablock',
  // Include the Python adapter in Pydocs' extraction inputs/cache key.
  search: ['../..', './node_modules/astro-myst-notebooks/dist'], docstringStyle: 'numpy',
  extensions: ['./griffe_extension.py'],
  sourceLink: { template: 'https://gitlab.kwant-project.org/qt/pymablock/-/blob/{ref}/{path}#L{start}-{end}', root: '../..', ref: 'main' },
  members: { exclude: ['pymablock.tests', 'pymablock.tests.*'] },
};
export const runner = { command: ['python', '-m', 'griffe'] };
// Use the checked-in inventories from the original documentation build.
export const references = Object.fromEntries(Object.entries({
  numpy: 'https://numpy.org/doc/stable/',
  scipy: 'https://docs.scipy.org/doc/scipy/',
  python: 'https://docs.python.org/3/',
  kwant: 'https://kwant-project.org/doc/1/',
  sympy: 'https://docs.sympy.org/dev/',
}).map(([id, base]) => [id, {
  file: new URL(`../source/_static/intersphinx-fallback/${id}.inv`, import.meta.url), base,
}]));
export const localInventory = new URL('./.astro/references/local.inv', import.meta.url);
export const referenceCache = new URL('./.astro/references/', import.meta.url);
export const execution = {
  cwd: new URL('../../', import.meta.url), timeout: 480,
  pixi: { manifest: new URL('../../pyproject.toml', import.meta.url), feature: 'starlight' },
};
export const interactive = {
  wheel: { project: new URL('../../', import.meta.url), command: ['python', '-m', 'hatchling', 'build', '-t', 'wheel'] },
  environment: new URL('./environment.yml', import.meta.url),
  startupTimeout: 180000,
};

export const documents = new URL('./.astro/references/documents.json', import.meta.url);
