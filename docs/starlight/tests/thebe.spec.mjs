import { test, expect } from '@playwright/test';

async function activate(page) {
  await page.locator('[data-thebe-controls]').getByRole('button', { name: 'Enable interactivity', exact: true }).click();
  await expect(page.locator('[data-thebe-controls]')).toHaveAttribute('data-state', 'ready', { timeout: 180000 });
}

async function replaceFirstCell(page, source) {
  await page.locator('.CodeMirror').first().click();
  await page.keyboard.press('ControlOrMeta+Home');
  await page.keyboard.press('ControlOrMeta+a');
  await page.keyboard.insertText(source);
}

test('Thebe runs real Python, accepts edits, clears old results, and resets the kernel', async ({ page }) => {
  test.setTimeout(240000);
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const downloads = [];
  page.on('request', request => downloads.push(request.url()));
  await page.goto('./tutorial/bilayer_graphene/');
  expect(downloads.some(url => url.includes('/thebe/'))).toBe(false);
  const sources = await page.locator('.jupyter-cell').evaluateAll(cells => cells.map(cell => cell.dataset.source));
  await activate(page);
  await expect(page.locator('.jupyter-outputs [data-mime="text/latex"]')).toHaveCount(4);
  await page.getByRole('button', { name: 'Run all', exact: true }).click();
  await expect(page.getByRole('status')).toHaveText('All cells completed.', { timeout: 60000 });
  await expect(page.locator('.jupyter-live-output [data-mime="text/latex"] .katex')).toHaveCount(4);
  await replaceFirstCell(page, 'print("Edited in browser")');
  await page.locator('.thebe-run-button').first().click();
  await expect(page.locator('.jupyter-live-output').first()).toContainText('Edited in browser');
  await replaceFirstCell(page, 'pass');
  await page.locator('.thebe-run-button').first().click();
  await expect(page.locator('.jupyter-live-output').first()).not.toContainText('Edited in browser');
  await expect(page.locator('.jupyter-outputs').first()).toBeHidden();
  await replaceFirstCell(page, sources[0]);
  await page.getByRole('button', { name: 'Restart Python', exact: true }).click();
  await expect(page.getByRole('status')).toContainText('Python restarted.', { timeout: 120000 });
  await page.getByRole('button', { name: 'Run all', exact: true }).click();
  await expect(page.getByRole('status')).toHaveText('All cells completed.', { timeout: 60000 });
  expect(errors).toEqual([]);
});

test('a runtime download failure preserves static content and supports retry', async ({ page }) => {
  test.setTimeout(240000);
  await page.route('**/thebe/index.js', route => route.abort());
  await page.goto('./tutorial/bilayer_graphene/');
  await page.locator('[data-thebe-controls]').getByRole('button', { name: 'Enable interactivity', exact: true }).click();
  await expect(page.locator('[data-thebe-controls]')).toHaveAttribute('data-state', 'error');
  await expect(page.locator('.jupyter-outputs [data-mime="text/latex"]')).toHaveCount(4);
  await page.unroute('**/thebe/index.js');
  await page.getByRole('button', { name: 'Retry interactivity', exact: true }).click();
  await expect(page.locator('[data-thebe-controls]')).toHaveAttribute('data-state', 'ready', { timeout: 180000 });
});

test('unchanged MyST source also runs in browser Python', async ({ page }) => {
  test.setTimeout(240000);
  await page.goto('./tutorial/getting_started/');
  await activate(page);
  await page.getByRole('button', { name: 'Run all', exact: true }).click();
  await expect(page.getByRole('status')).toContainText('All cells completed.', { timeout: 120000 });
  await expect(page.locator('.jupyter-live-output img').first()).toBeVisible();
  await expect(page.locator('.jupyter-cell[hidden]')).toHaveCount(1);
});

test('notebook removal shuts down its session and reconnects without reloading the page', async ({ page }) => {
  test.setTimeout(240000);
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('./tutorial/bilayer_graphene/');
  await activate(page);
  await page.evaluate(() => {
    window.oldServer = window.thebe.server;
    window.oldSession = window.thebe.notebook.session;
    window.savedNotebook = document.querySelector('jupyter-notebook');
    window.savedNotebookParent = window.savedNotebook.parentElement;
    window.savedNotebook.remove();
  });
  await expect.poll(() => page.evaluate(() => window.oldServer.isDisposed)).toBe(true);
  expect(await page.evaluate(() => window.oldSession.kernel?.isDisposed ?? true)).toBe(true);
  await page.evaluate(() => window.savedNotebookParent.append(window.savedNotebook));
  await activate(page);
  await page.getByRole('button', { name: 'Run all', exact: true }).click();
  await expect(page.getByRole('status')).toHaveText('All cells completed.', { timeout: 60000 });
  expect(await page.evaluate(() => window.oldSession !== window.thebe.notebook.session)).toBe(true);
  expect(errors).toEqual([]);
});

test('single-cell execution permits reset; Run all stops on errors and recovers', async ({ page }) => {
  test.setTimeout(240000);
  await page.goto('./tutorial/bilayer_graphene/');
  await activate(page);
  const original = await page.locator('.jupyter-cell').first().getAttribute('data-source');
  await replaceFirstCell(page, 'import time\ntime.sleep(2)\nprint("finished")');
  await page.locator('.thebe-run-button').first().click();
  await expect(page.getByRole('button', { name: 'Restart Python' })).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Run all', exact: true })).toBeDisabled();
  await expect(page.locator('.jupyter-live-output').first()).toContainText('finished');
  await expect(page.getByRole('button', { name: 'Restart Python' })).toBeEnabled();
  await replaceFirstCell(page, 'raise ValueError("reader mistake")');
  await page.getByRole('button', { name: 'Run all', exact: true }).click();
  await expect(page.getByRole('status')).toContainText('A cell failed');
  await expect(page.locator('.jupyter-live-output').nth(1)).toBeEmpty();
  await replaceFirstCell(page, original);
  await page.getByRole('button', { name: 'Run all', exact: true }).click();
  await expect(page.getByRole('status')).toHaveText('All cells completed.', { timeout: 60000 });
});

test('removing a notebook during startup cannot attach a late session', async ({ page }) => {
  test.setTimeout(240000);
  await page.goto('./tutorial/bilayer_graphene/');
  // Hold the first runtime download while the component is removed.
  let release;
  const held = new Promise(resolve => { release = resolve; });
  await page.route('**/thebe/index.js', async route => { await held; await route.continue(); });
  await page.locator('[data-thebe-controls]').getByRole('button', { name: 'Enable interactivity', exact: true }).click();
  await expect(page.locator('[data-thebe-controls]')).toHaveAttribute('data-state', 'loading');
  await page.evaluate(() => document.querySelector('jupyter-notebook').remove());
  release();
  await expect.poll(() => page.evaluate(() => Boolean(window.thebe?.bootstrap))).toBe(true);
  expect(await page.evaluate(() => window.thebe.server)).toBeUndefined();
});

test('startup timeout restores the static notebook', async ({ page }) => {
  await page.clock.install();
  let release;
  const held = new Promise(resolve => { release = resolve; });
  await page.route('**/thebe/index.js', async route => { await held; await route.abort(); });
  await page.goto('./tutorial/bilayer_graphene/');
  await page.locator('[data-thebe-controls]').getByRole('button', { name: 'Enable interactivity', exact: true }).click();
  await expect(page.locator('[data-thebe-controls]')).toHaveAttribute('data-state', 'loading');
  await page.clock.fastForward(180001);
  await expect(page.locator('[data-thebe-controls]')).toHaveAttribute('data-state', 'error');
  await expect(page.locator('.jupyter-outputs [data-mime="text/latex"]')).toHaveCount(4);
  release();
});


test('reset terminates an infinite Python loop and preserves edited source', async ({ page }) => {
  test.setTimeout(240000);
  const errors = [];
  const workers = new Set();
  page.on('worker', worker => { workers.add(worker); worker.on('close', () => workers.delete(worker)); });
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('./tutorial/bilayer_graphene/');
  await activate(page);
  const originalWorkers = [...workers];
  expect(originalWorkers.length).toBeGreaterThan(0);
  await page.evaluate(() => { window.oldKernel = window.thebe.notebook.session.kernel; });
  await replaceFirstCell(page, 'while True: pass');
  await page.getByRole('button', { name: 'Run all', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Restart Python' })).toBeEnabled();
  await page.getByRole('button', { name: 'Restart Python' }).click();
  await expect(page.getByRole('status')).toContainText('Python restarted.', { timeout: 120000 });
  expect(await page.evaluate(() => window.oldKernel.isDisposed)).toBe(true);
  await expect.poll(() => originalWorkers.filter(worker => workers.has(worker)).length).toBe(0);
  await replaceFirstCell(page, 'print("recovered after infinite loop")');
  await page.locator('.thebe-run-button').first().click();
  await expect(page.locator('.jupyter-live-output').first()).toContainText('recovered after infinite loop');
  expect(errors).toEqual([]);
});

test('a stalled session cleanup is bounded and retry starts usable Python', async ({ page }) => {
  test.setTimeout(240000);
  await page.goto('./tutorial/bilayer_graphene/');
  await activate(page);
  await page.evaluate(() => { window.thebe.notebook.session.shutdown = () => new Promise(() => {}); });
  await page.getByRole('button', { name: 'Restart Python' }).click();
  await expect(page.locator('[data-thebe-controls]')).toHaveAttribute('data-state', 'error', { timeout: 10000 });
  await page.getByRole('button', { name: 'Retry interactivity' }).click();
  await expect(page.locator('[data-thebe-controls]')).toHaveAttribute('data-state', 'ready', { timeout: 120000 });
  await replaceFirstCell(page, 'print("cleanup recovered")');
  await page.locator('.thebe-run-button').first().click();
  await expect(page.locator('.jupyter-live-output').first()).toContainText('cleanup recovered');
});


test('removal during package loading terminates the allocated Python worker', async ({ page, context }) => {
  test.setTimeout(180000);
  const workers = new Set();
  page.on('worker', worker => { workers.add(worker); worker.on('close', () => workers.delete(worker)); });
  let release;
  let blocked = 0;
  const held = new Promise(resolve => { release = resolve; });
  await context.route('**/thebe/xeus/**/empack_env_meta.json', async route => {
    blocked++;
    await held;
    await route.abort().catch(() => {});
  });
  try {
    await page.goto('./tutorial/bilayer_graphene/');
    await page.locator('[data-thebe-controls]').getByRole('button', { name: 'Enable interactivity', exact: true }).click();
    await expect.poll(() => blocked, { timeout: 120000 }).toBeGreaterThan(0);
    expect(workers.size).toBeGreaterThan(0);
    await page.evaluate(() => { window.removedServer = window.thebe.server; document.querySelector('jupyter-notebook').remove(); });
    await expect.poll(() => workers.size, { timeout: 10000 }).toBe(0);
    await expect.poll(() => page.evaluate(() => window.removedServer.isDisposed)).toBe(true);
  } finally { release(); }
});
