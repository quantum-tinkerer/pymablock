import { api, runner, references, localInventory, referenceCache, execution, interactive, documents } from './docs.config.mjs';
import { copyFile, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightPydocs, { pydocsSidebarGroup } from 'starlight-pydocs';
import notebooks from 'astro-myst-notebooks';
import { exportInventory } from 'astro-myst-notebooks/inventory';
import { site, base } from './deployment.mjs';
import { sidebar, sources } from './navigation.mjs';

export default defineConfig({
  site, base,
  integrations: [notebooks({ execution, interactive, references, localInventory, referenceCache, documents }), starlight({
    title: 'Pymablock',
    description: 'Effective models, order by order.',
    logo: { light: '../source/_static/logo.svg', dark: '../source/_static/logo_dark.svg', replacesTitle: false },
    favicon: '/favicon.png',
    social: [{ icon: 'gitlab', label: 'GitLab', href: 'https://gitlab.kwant-project.org/qt/pymablock' }],
    customCss: ['./src/styles/custom.css'],
    routeMiddleware: './src/api-toc.ts',
    components: { MarkdownContent: './src/components/MarkdownContent.astro', Footer: './src/components/Footer.astro' },
    plugins: [starlightPydocs({
      packages: [api], runner,
      components: { Signature: './src/components/ApiSignature.astro', DocstringSections: 'astro-myst-notebooks/CheckedDocstrings.astro' },
      inventories: Object.values(references).map(({ file, base }) => ({ file: fileURLToPath(file), base })),
    })],
    sidebar: [...sidebar, { label: 'API by module', items: [pydocsSidebarGroup], collapsed: true }],
  }), {
    name: 'pymablock-api-inventory',
    hooks: {
      'astro:build:done': async ({ dir }) => {
        await exportInventory({ api: localInventory, documents, base, root: execution.cwd, destination: new URL('objects.inv', dir),
          labels: Object.fromEntries(['genindex', 'modindex', 'py-modindex'].map(name => [name, {
            location: `${base.replace(/\/$/, '')}/api/pymablock/`, display: 'API index',
          }]).concat([['search', {location: `${base.replace(/\/$/, '')}/`, display: 'Search'}]])),
        });
        for (const file of Object.keys(sources)) {
          const destination = new URL(`_sources/${file}`, dir);
          await mkdir(new URL('./', destination), { recursive: true });
          await copyFile(new URL(`../source/${file}`, import.meta.url), destination);
        }
      },
    },
  }],
});
