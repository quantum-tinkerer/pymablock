import { test, expect } from '@playwright/test';

test('page actions sit beside the repository link and work on mobile', async ({ page }) => {
  await page.goto('./');
  const header = page.locator('header');
  await expect(header.getByRole('link', {name: 'GitLab', exact: true})).toBeVisible();
  const source = header.getByRole('link', {name: 'Download MyST source'});
  await expect(source).toBeVisible();
  await expect(source).toHaveAttribute('href', /\/_sources\/index.md$/);
  await expect(header.getByRole('link', {name: 'Report an issue'})).toBeVisible();
  await page.evaluate(() => { window.print = () => { window.printCalled = true; }; });
  await header.getByRole('button', {name: 'Print / save PDF'}).click();
  expect(await page.evaluate(() => window.printCalled)).toBe(true);
  const [download] = await Promise.all([page.waitForEvent('download'), source.click()]);
  expect(download.suggestedFilename()).toBe('index.md');
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('button', {name: 'Menu', exact: true}).click();
  await expect(page.getByRole('link', {name: 'Download MyST source'}).filter({visible:true})).toBeVisible();
  await expect(page.getByRole('button', {name: 'Print / save PDF'}).filter({visible:true})).toBeVisible();
  await expect(page.getByRole('link', {name: 'Report an issue'}).filter({visible:true})).toBeVisible();
  await page.goto('./api/pymablock/block_diagonalization/');
  await expect(page.getByRole('link', {name: 'Download MyST source'})).toHaveCount(0);
});
