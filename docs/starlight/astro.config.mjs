import { api, runner, references, localInventory, referenceCache, execution, interactive, documents } from './docs.config.mjs';
import { copyFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightPydocs, { pydocsSidebarGroup } from 'starlight-pydocs';
import notebooks from 'astro-myst-notebooks';
import { site, base } from './deployment.mjs';
import { sidebar, sources } from './navigation.mjs';

export default defineConfig({
  site, base,
  integrations: [notebooks({ execution, interactive, references, localInventory, referenceCache, documents }), starlight({
    title: 'Pymablock',
    description: 'Effective models, order by order.',
    logo: { src: './src/assets/logo.svg', replacesTitle: false },
    favicon: '/favicon.png',
    social: [{ icon: 'gitlab', label: 'GitLab', href: 'https://gitlab.kwant-project.org/qt/pymablock' }],
    customCss: ['./src/styles/custom.css'],
    plugins: [starlightPydocs({
      packages: [api], runner,
      inventories: Object.values(references).map(({ file, base }) => ({ file: fileURLToPath(file), base })),
    })],
    sidebar: [...sidebar, { label: 'API by module', items: [pydocsSidebarGroup], collapsed: true }],
  }), {
    name: 'pymablock-api-inventory',
    hooks: {
      'astro:build:done': async ({ dir }) => {
        await copyFile(localInventory, new URL('objects.inv', dir));
        // Some static hosts prefer name.html over name/index.html. A full-page
        // alias preserves both URL forms without a redirect loop on those hosts.
        for (const { id } of Object.values(sources)) {
          if (id !== 'index') await copyFile(new URL(`${id}/index.html`, dir), new URL(`${id}.html`, dir));
        }
      },
    },
  }],
});
