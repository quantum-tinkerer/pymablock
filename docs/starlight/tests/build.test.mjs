import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, readFile, writeFile, unlink, rm, mkdir } from 'node:fs/promises';
import { join, resolve } from 'node:path';

function build(output) {
  return new Promise((resolve, reject) => {
    const child = spawn('npm', ['run', 'build', '--', '--outDir', output], { stdio: ['ignore', 'pipe', 'pipe'] });
    let log = '';
    child.stdout.on('data', chunk => { log += chunk; });
    child.stderr.on('data', chunk => { log += chunk; });
    child.on('error', reject);
    child.on('close', code => resolve({ code, log }));
  });
}

test('real Astro builds reexecute unchanged pages, propagate failures, and recover', async () => {
  await mkdir('.astro', {recursive:true});
  const directory = await mkdtemp(join(resolve('.astro'), 'build-test-'));
  const output = join(directory, 'html');
  const counter = join(directory, 'count');
  const fixture = '../source/build-contract-check.md';
  const source = `---\ntitle: Build contract\n---\n\n\`\`\`{code-cell} ipython3\nfrom pathlib import Path\np = Path(${JSON.stringify(counter)})\np.write_text(str(int(p.read_text()) + 1) if p.exists() else "1")\n\`\`\`\n`;
  await writeFile(fixture, source, { flag: 'wx' });
  try {
    for (const expected of [1, 2]) {
      const result = await build(output);
      assert.equal(result.code, 0, result.log);
      assert.equal(Number(await readFile(counter, 'utf8')), expected, 'execute once per page on every build');
    }
    await writeFile(fixture, `${source}\n\`\`\`{code-cell} ipython3\nraise RuntimeError("intentional-build-failure")\n\`\`\`\n`);
    const failed = await build(output);
    assert.notEqual(failed.code, 0, 'cell failure must fail Astro, not just log a warning');
    assert.match(failed.log, /intentional-build-failure/);
  } finally {
    await unlink(fixture);
    const restored = await build(output);
    await rm(directory, { recursive: true });
    assert.equal(restored.code, 0, restored.log);
  }
});
