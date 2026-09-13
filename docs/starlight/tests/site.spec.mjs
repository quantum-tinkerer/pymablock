import { test, expect } from '@playwright/test';
import { sources } from '../navigation.mjs';
import { readFile, readdir } from 'node:fs/promises';
import { resolve } from 'node:path';

for (const [file, {id, title}] of Object.entries(sources)) {
  test(`complete source page: ${file}`, async ({page}) => {
    const route = id === 'index' ? '/' : `/${id}/`;
    await page.goto('.' + route);
    await expect(page.locator('h1')).toHaveCount(1);
    await expect(page.locator('h1').first()).toHaveText(title);
    await expect(page.locator('.katex-error, .unhandled')).toHaveCount(0);
    const source = await readFile(resolve('../source', file), 'utf8');
    const count = [...source.matchAll(/^`{3,}\{code-cell\}/gm)].length;
    await expect(page.locator('.jupyter-cell')).toHaveCount(count);
    await expect(page.getByRole('button', {name:'Run interactively',exact:true})).toHaveCount(count ? 1 : 0);
    for (const image of await page.locator('main img').all()) {
      expect(await image.evaluate(img => img.complete && img.naturalWidth > 0)).toBe(true);
    }
  });
}

test('all built pages have valid local links, anchors, and assets', async ({page, request}) => {
  const files = (await readdir('dist', {recursive:true})).filter(name=>name.endsWith('index.html'));
  const pages = new Map();
  for (const file of files) {
    const route = '/' + file.replace(/index\.html$/, '');
    await page.goto('.' + route);
    pages.set(new URL(route.slice(1), test.info().project.use.baseURL).pathname, await page.evaluate(()=>({
      ids: [...document.querySelectorAll('[id]')].map(el=>el.id),
      links: [...document.querySelectorAll('main a[href]')].map(el=>el.getAttribute('href')),
    })));
  }
  const failures=[];
  for (const [route,{links}] of pages) for (const href of links) {
    const url = new URL(href, 'https://docs.test'+route);
    if(url.origin!=='https://docs.test') continue;
    const destination = pages.get(url.pathname) ?? pages.get(url.pathname+'/');
    if(destination) {
      if(url.hash && !destination.ids.includes(url.hash.slice(1)) && !destination.ids.includes(decodeURIComponent(url.hash.slice(1)))) failures.push(`${route}: missing ${href}`);
    } else if(!(await request.get(url.pathname)).ok()) failures.push(`${route}: missing ${href}`);
  }
  expect([...new Set(failures)]).toEqual([]);
});

test('mobile layout and API descriptions remain readable in both themes', async ({page}) => {
  await page.setViewportSize({width:390,height:844});
  for(const colorScheme of ['light','dark']) {
    await page.emulateMedia({colorScheme});
    for(const {id} of Object.values(sources)) {
      const route=id==='index'?'/':`/${id}/`;
      await page.goto('.' + route);
      expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),route).toBe(true);
    }
    await page.goto('./api/pymablock/block_diagonalization/');
    const widths=await page.locator('.pyd-param-description').first().evaluate(el=>({description:el.getBoundingClientRect().width,table:el.closest('table').getBoundingClientRect().width}));
    expect(widths.description).toBeGreaterThan(widths.table*.95);
  }
  await page.getByRole('button',{name:'Menu',exact:true}).click();
  await expect(page.getByRole('link',{name:'Getting started',exact:true}).first()).toBeVisible();
});

test('search indexes original documentation and API objects', async ({page}) => {
  await page.goto('./');
  await page.getByRole('button',{name:'Search'}).click();
  await page.getByRole('textbox', {name:'Search', exact:true}).fill('block_diagonalize');
  await expect(page.locator('dialog')).toContainText(/\d+ results for block_diagonalize/);
  await expect(page.locator('dialog a[href$="/tutorial/getting_started/"]').first()).toBeVisible();
});

test('legacy page aliases and the API inventory remain available', async ({page, request}) => {
  await page.goto('./tutorial/getting_started.html');
  await expect(page.locator('h1').first()).toHaveText('Getting started');
  const inventory = await request.get('objects.inv');
  expect(inventory.ok()).toBe(true);
  expect((await inventory.body()).subarray(0, 33).toString()).toContain('# Sphinx inventory version 2');
});
