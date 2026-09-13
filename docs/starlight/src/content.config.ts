import { api, runner, localInventory, documents } from '../docs.config.mjs';
import { sources } from '../navigation.mjs';
import { pydocsInventory } from 'astro-myst-notebooks/pydocs';
import { defineCollection } from 'astro:content';
import { sourceLoader } from 'astro-myst-notebooks/loader';
import { docsSchema } from '@astrojs/starlight/schema';

export const collections = {
  apiSymbols: defineCollection({ loader: pydocsInventory({ ...api, runner }, localInventory) }),
  docs: defineCollection({
    loader: sourceLoader({
      base: new URL('../../source/', import.meta.url),
      pattern: '**/[^_]*.md',
      sources: Object.fromEntries(Object.entries(sources).map(([file, source]) => [file, {
        ...source,
        editUrl: `https://gitlab.kwant-project.org/qt/pymablock/-/edit/main/docs/source/${file}`,
      }])),
      documents,
    }),
    schema: docsSchema(),
  }),
};
