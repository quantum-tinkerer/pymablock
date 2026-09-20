import { test, expect } from '@playwright/test';
import { sources } from '../navigation.mjs';

// These execute the actual source cells, including the full-size Kwant system.
for (const { id } of Object.values(sources).filter(({ id }) => id.startsWith('tutorial/') || ['algorithms', 'radius'].includes(id))) {
  test(`unchanged browser notebook: ${id}`, async ({ page }) => {
    test.setTimeout(600000);
    await page.goto(`./${id}/`);
    await page.locator('[data-thebe-controls]').getByRole('button', { name: 'Enable interactivity', exact: true }).click();
    await expect(page.locator('[data-thebe-controls]')).toHaveAttribute('data-state', 'ready', { timeout: 180000 });
    await page.getByRole('button', { name: 'Run all', exact: true }).click();
    await page.waitForFunction(() => ['ready', 'error'].includes(document.querySelector('[data-thebe-controls]')?.dataset.state), {}, { timeout: 480000 });
    const outputs = await page.locator('.jupyter-live-output').allTextContents();
    expect(await page.getByRole('status').textContent(), outputs.join('\n').slice(-10000)).toBe('All cells completed.');
    await expect(page.locator('.jupyter-live-output .katex-error')).toHaveCount(0);
    await expect(page.locator('.CodeMirror')).toHaveCount(await page.locator('.jupyter-cell').count());
  });
}
