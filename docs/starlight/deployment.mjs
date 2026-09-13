// Read the Docs provides the complete version URL, including custom domains.
const canonical = process.env.READTHEDOCS_CANONICAL_URL;
const url = new URL(process.env.DOCS_SITE || canonical || 'https://pymablock.readthedocs.io/');
export const site = url.origin;
export const base = process.env.DOCS_BASE || (canonical ? url.pathname : '/');
