import { test, expect } from '@playwright/test';

test('equation numbers and subscripts use matching KaTeX styles', async ({ page }) => {
  await page.goto('./');
  await page.evaluate(() => document.fonts.ready);
  const equations = page.locator('.katex-display .katex-html');
  expect(await equations.count()).toBeGreaterThanOrEqual(2);
  for (const equation of (await equations.all()).slice(0, 2)) {
    const bounds = await equation.evaluate(element => {
      const tag = element.querySelector('.katex-tag');
      const base = element.querySelector(':scope > .katex-base');
      return { position: tag && getComputedStyle(tag).position,
        gap: tag.getBoundingClientRect().left - base.getBoundingClientRect().right,
        sub: element.querySelector('.msupsub')?.getBoundingClientRect().height };
    });
    expect(bounds.position).toBe('absolute');
    expect(bounds.gap).toBeGreaterThan(8);
    expect(bounds.sub).toBeGreaterThan(10);
  }
  await page.locator('main').screenshot({path: test.info().outputPath('equations.png')});
});
