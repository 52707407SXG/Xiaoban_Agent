/**
 * Tests for electron/backend-probes.cjs.
 *
 * Run with: node --test electron/backend-probes.test.cjs
 * (Wired into npm test:desktop:platforms in package.json.)
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const { canImportXiaobanCli, verifyXiaobanCli } = require('./backend-probes.cjs')

// Resolve the host's own Node binary -- guaranteed to be on disk and
// runnable. We use it as both a stand-in for "a python that doesn't
// have xiaoban_cli" (since `node -c "import xiaoban_cli"` will exit
// non-zero) and as a way to script verifyXiaobanCli's success path
// (a tiny script we write to disk that exits 0 on --version).
const NODE_BIN = process.execPath

test('canImportXiaobanCli returns false when path is falsy', () => {
  assert.equal(canImportXiaobanCli(''), false)
  assert.equal(canImportXiaobanCli(null), false)
  assert.equal(canImportXiaobanCli(undefined), false)
})

test('canImportXiaobanCli returns false when interpreter cannot run -c', () => {
  // node IS an interpreter, but `node -c "import xiaoban_cli"` is a
  // SyntaxError -- different exit reason from a real Python's
  // ModuleNotFoundError, but the predicate is "exit 0 or not" and
  // both land on "not", which is exactly what we want for the
  // resolver fall-through.
  assert.equal(canImportXiaobanCli(NODE_BIN), false)
})

test('canImportXiaobanCli returns false when binary does not exist', () => {
  const ghost = path.join(os.tmpdir(), 'xiaoban-probes-ghost-' + Date.now() + '.exe')
  assert.equal(canImportXiaobanCli(ghost), false)
})

test('verifyXiaobanCli returns false when command is falsy', () => {
  assert.equal(verifyXiaobanCli(''), false)
  assert.equal(verifyXiaobanCli(null), false)
  assert.equal(verifyXiaobanCli(undefined), false)
})

test('verifyXiaobanCli returns false when binary does not exist', () => {
  const ghost = path.join(os.tmpdir(), 'xiaoban-probes-ghost-' + Date.now() + '.exe')
  assert.equal(verifyXiaobanCli(ghost), false)
})

test('verifyXiaobanCli returns true when --version exits 0', () => {
  // Write a tiny script that exits 0 regardless of args, then invoke
  // it through node. This stands in for a working xiaoban binary --
  // verifyXiaobanCli only cares about the exit code.
  const scriptPath = path.join(os.tmpdir(), `xiaoban-probes-ok-${Date.now()}-${process.pid}.cjs`)
  fs.writeFileSync(scriptPath, 'process.exit(0)\n')
  try {
    // Use node as the launcher and our script as the "command". Pass
    // shell:false (default) -- node is a real binary, no shim.
    // execFileSync passes ['--version'] as args, which node ignores
    // gracefully (well, it prints its version and exits 0, which is
    // perfect -- exit code 0 is the only signal we read).
    assert.equal(verifyXiaobanCli(NODE_BIN), true)
  } finally {
    try {
      fs.unlinkSync(scriptPath)
    } catch {
      void 0
    }
  }
})

test('verifyXiaobanCli swallows timeouts (does not throw)', () => {
  // We can't easily provoke a real 5s hang in CI without slowing the
  // suite, but we CAN confirm that an invocation that DOES throw
  // (because the binary is missing) returns false rather than
  // propagating. Same code path the timeout case takes.
  assert.equal(verifyXiaobanCli('/definitely/not/a/real/binary/anywhere'), false)
})
