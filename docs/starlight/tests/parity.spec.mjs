import { test, expect } from '@playwright/test';
import { inflateSync } from 'node:zlib';
import { readFile } from 'node:fs/promises';
const fixture = JSON.parse(await readFile(new URL('./fixtures/sphinx-parity.json', import.meta.url)));
const aliases = JSON.parse(await readFile(new URL('../legacy-anchors.json', import.meta.url)));
const normalize = value => value.toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');

for (const [id, expected] of Object.entries(fixture.pages)) {
  test(`Sphinx prose and equation parity: ${id}`, async ({page}) => {
    await page.goto(id === 'index' ? './' : `./${id}/`);
    const text = await page.locator('.sl-markdown-content').evaluate(el => {
      const copy = el.cloneNode(true);
      copy.querySelectorAll('.katex, pre, .jupyter-outputs').forEach(node=>node.remove());
      return copy.textContent;
    });
    const missing = expected.prose.filter(prose => !normalize(text).includes(normalize(prose)));
    expect(missing).toEqual([]);
    for (const [alias, number] of Object.entries(expected.equations)) {
      const target = aliases[id][alias];
      const actual = await page.locator(`[id="${target}"] .tag`).textContent();
      expect(normalize(actual)).toBe(normalize(number));
    }
    for (const alias of Object.keys(aliases[id] ?? {})) await expect(page.locator(`[id="${alias}"]`)).toHaveCount(1);
  });
}

test('complete inline API reference matches the original documented objects and prose', async ({page}) => {
  await page.goto('./documentation/pymablock/');
  const missing = [];
  for (const [name, descriptions] of Object.entries(fixture.objects)) {
    const member = page.locator(`[data-pydocs-path="${name}"]`);
    await expect(member).toHaveCount(1);
    await expect(page.locator(`starlight-toc a[href="#${name}"]`)).toHaveCount(1);
    const text = await member.evaluate(el => {
      const copy=el.cloneNode(true);
      copy.querySelectorAll('.katex,.pyd-signature,.pyd-group,.pyd-provenance').forEach(node=>node.remove());
      return copy.textContent;
    });
    for (const prose of descriptions) if(!normalize(text).includes(normalize(prose))) missing.push({name, prose});
  }
  expect(missing).toEqual([]);
  await expect(page.locator('template[data-myst-autodoc]')).toHaveCount(0);
  await expect(page.locator('.pyd-member [data-source-location*="document.md"]')).toHaveCount(0);
  await expect(page.locator('[data-pydocs-signature="pymablock.number_ordered_form.NumberOrderedForm"]')).toContainText('validate');
  await expect(page.locator('[data-pydocs-path="pymablock.series.BlockSeries"] > [data-pydocs-section="parameters"] .pyd-param-name')).toHaveCount(6);
});

test('source, edit, issue and print actions are available', async ({page,request}) => {
  await page.goto('./tutorial/getting_started/');
  const source=page.getByRole('link',{name:'Download MyST source'});
  expect(await (await request.get(await source.getAttribute('href'))).text()).toBe(await readFile('../source/tutorial/getting_started.md','utf8'));
  await expect(page.getByRole('link',{name:'Edit page'})).toHaveAttribute('href',/\/edit\/main\/docs\/source\/tutorial\/getting_started.md$/);
  await expect(page.getByRole('link',{name:'Report an issue'})).toBeVisible();
  await expect(page.getByRole('button',{name:'Print / save PDF'})).toBeVisible();
});

test('exported inventory retains every original object type and document', async ({page, request}) => {
  const buffer = await (await request.get('objects.inv')).body();
  let offset = 0;
  for(let i=0;i<4;i++) offset=buffer.indexOf(10,offset)+1;
  const records=inflateSync(buffer.subarray(offset)).toString().trim().split('\n').map(line=>line.split(/\s+/));
  const actual=new Set(records.map(([name,type])=>`${type}:${name}`));
  expect(fixture.inventory.filter(({name,type})=>!actual.has(`${type}:${name}`))).toEqual([]);
  const pages = new Map();
  for (const [name, , , rawTarget] of records) {
    const target = rawTarget.replace(/\$$/, name);
    const [path, fragment] = target.split('#');
    if (!pages.has(path)) {
      const response = await request.get(path || './');
      expect(response.ok(), target).toBe(true);
      pages.set(path, await page.evaluate(html => {
        const document = new DOMParser().parseFromString(html, 'text/html');
        return [...document.querySelectorAll('[id]')].map(node => node.id);
      }, await response.text()));
    }
    if (fragment) expect(pages.get(path), target).toContain(decodeURIComponent(fragment));
  }
});
