const assert = require('node:assert/strict')
const { test } = require('node:test')

const {
  expandWindowsEnvRefs,
  parseRegQueryValue,
  readWindowsUserEnvVar
} = require('./windows-user-env.cjs')

// ── parseRegQueryValue ─────────────────────────────────────────────────────

test('parseRegQueryValue extracts a REG_SZ value', () => {
  const out = [
    '',
    'HKEY_CURRENT_USER\\Environment',
    '    XIAOBAN_HOME    REG_SZ    F:\\Xiaoban\\data',
    ''
  ].join('\r\n')
  assert.equal(parseRegQueryValue(out, 'XIAOBAN_HOME'), 'F:\\Xiaoban\\data')
})

test('parseRegQueryValue matches the name case-insensitively', () => {
  const out = 'HKEY_CURRENT_USER\\Environment\r\n    Xiaoban_Home    REG_EXPAND_SZ    %USERPROFILE%\\h\r\n'
  assert.equal(parseRegQueryValue(out, 'XIAOBAN_HOME'), '%USERPROFILE%\\h')
})

test('parseRegQueryValue preserves spaces inside the value', () => {
  const out = '    XIAOBAN_HOME    REG_SZ    C:\\Program Files\\Xiaoban\r\n'
  assert.equal(parseRegQueryValue(out, 'XIAOBAN_HOME'), 'C:\\Program Files\\Xiaoban')
})

test('parseRegQueryValue returns null when the value line is absent', () => {
  const out = 'HKEY_CURRENT_USER\\Environment\r\n    Path    REG_SZ    C:\\x\r\n'
  assert.equal(parseRegQueryValue(out, 'XIAOBAN_HOME'), null)
  assert.equal(parseRegQueryValue('', 'XIAOBAN_HOME'), null)
  assert.equal(parseRegQueryValue('garbage', 'XIAOBAN_HOME'), null)
})

// ── expandWindowsEnvRefs ───────────────────────────────────────────────────

test('expandWindowsEnvRefs expands %VAR% case-insensitively', () => {
  assert.equal(
    expandWindowsEnvRefs('%UserProfile%\\h', { USERPROFILE: 'C:\\Users\\jeff' }),
    'C:\\Users\\jeff\\h'
  )
})

test('expandWindowsEnvRefs leaves literal paths and unknown refs intact', () => {
  assert.equal(expandWindowsEnvRefs('F:\\Xiaoban\\data', {}), 'F:\\Xiaoban\\data')
  assert.equal(expandWindowsEnvRefs('%NOPE%\\x', {}), '%NOPE%\\x')
})

// ── readWindowsUserEnvVar ──────────────────────────────────────────────────

test('readWindowsUserEnvVar returns null off Windows without spawning', () => {
  let spawned = false
  const exec = () => {
    spawned = true
    return ''
  }
  assert.equal(readWindowsUserEnvVar('XIAOBAN_HOME', { platform: 'linux', exec }), null)
  assert.equal(spawned, false)
})

test('readWindowsUserEnvVar queries HKCU\\Environment and expands the value', () => {
  const calls = []
  const exec = (cmd, args) => {
    calls.push([cmd, args])
    return 'HKEY_CURRENT_USER\\Environment\r\n    XIAOBAN_HOME    REG_EXPAND_SZ    %DRIVE%\\Xiaoban\r\n'
  }
  const value = readWindowsUserEnvVar('XIAOBAN_HOME', {
    platform: 'win32',
    env: { DRIVE: 'F:' },
    exec
  })
  assert.equal(value, 'F:\\Xiaoban')
  assert.deepEqual(calls, [['reg', ['query', 'HKCU\\Environment', '/v', 'XIAOBAN_HOME']]])
})

test('readWindowsUserEnvVar returns null when reg exits non-zero (value missing)', () => {
  const exec = () => {
    throw new Error('reg exited 1')
  }
  assert.equal(readWindowsUserEnvVar('XIAOBAN_HOME', { platform: 'win32', exec }), null)
})

test('readWindowsUserEnvVar returns null for an empty value', () => {
  const exec = () => '    XIAOBAN_HOME    REG_SZ    \r\n'
  assert.equal(readWindowsUserEnvVar('XIAOBAN_HOME', { platform: 'win32', exec }), null)
})
