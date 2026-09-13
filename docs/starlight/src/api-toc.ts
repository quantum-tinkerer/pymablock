import { defineRouteMiddleware } from '@astrojs/starlight/route-data';
import { getCollection } from 'astro:content';
import { autodocOutline } from 'astro-myst-notebooks/autodoc';

export const onRequest = defineRouteMiddleware(async ({ locals }) => {
  const route = locals.starlightRoute;
  if (!route.toc || !route.entry.body?.includes('{autodoc}')) return;
  const symbols = (await getCollection('apiSymbols')).map(entry => entry.data);
  const byPath = new Map(symbols.map(symbol => [symbol.path, symbol]));
  type Item = NonNullable<typeof route.toc>['items'][number];
  const flatten = (items: Item[]): Item[] => items.flatMap(item => [item, ...flatten(item.children)]);
  const sections = flatten(route.toc.items);
  const itemFor = (path: string, depth: number): Item => {
    const symbol = byPath.get(path);
    if (!symbol) throw new Error(`Unknown API table-of-contents object: ${path}`);
    const children = symbols
      .filter(other => other.path.slice(0, other.path.lastIndexOf('.')) === path)
      .sort((a, b) => a.name.localeCompare(b.name));
    return {
      depth,
      slug: path,
      text: symbol.name,
      children: children.map(child => itemFor(child.path, depth + 1)),
    };
  };
  for (const { section, name } of autodocOutline(route.entry.body)) {
    const parent = sections.find(item => item.text === section);
    if (!parent) throw new Error(`API documentation needs a section heading before ${name}`);
    parent.children.push(itemFor(name, parent.depth + 1));
  }
  route.toc.maxHeadingLevel = 4;
  route.headings = flatten(route.toc.items).filter(item => item.slug !== '_top');
});
