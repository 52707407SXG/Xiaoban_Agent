import { PassThrough } from 'stream'

import { renderSync } from '@xiaoban/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { SessionPanel } from '../components/branding.js'
import { DEFAULT_THEME } from '../theme.js'
import type { McpServerStatus, SessionInfo } from '../types.js'

// Startup branding should stay focused on the Xiaoban welcome card. Tool,
// skill, and MCP details belong behind commands/diagnostics, not on the first
// screen.

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

const makeStreams = (columns = 100) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()

  Object.assign(stdout, { columns, isTTY: false, rows: 40 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })

  let captured = ''
  stdout.on('data', chunk => {
    captured += chunk.toString()
  })

  return { capture: () => captured, stderr, stdin, stdout }
}

const mcp = (over: Partial<McpServerStatus> & Pick<McpServerStatus, 'name'>): McpServerStatus => ({
  connected: false,
  tools: 0,
  transport: 'http',
  ...over
})

const baseInfo = (mcp_servers: McpServerStatus[]): SessionInfo => ({
  mcp_servers,
  model: 'test-model',
  skills: { core: ['a', 'b'] },
  tools: { file: ['read_file', 'write_file'] }
})

async function renderFrame(info: SessionInfo): Promise<string> {
  const streams = makeStreams()

  const instance = renderSync(React.createElement(SessionPanel, { info, sid: 'test', t: DEFAULT_THEME }), {
    patchConsole: false,
    stderr: streams.stderr as NodeJS.WriteStream,
    stdin: streams.stdin as NodeJS.ReadStream,
    stdout: streams.stdout as NodeJS.WriteStream
  })

  try {
    await delay(20)

    // Strip ANSI so we can assert on the rendered text content.
    // eslint-disable-next-line no-control-regex
    return streams.capture().replace(/\u001b\[[0-9;]*m/g, '')
  } finally {
    instance.unmount()
    instance.cleanup()
  }
}

describe('startup branding card', () => {
  it('shows the Xiaoban card without tool or MCP counters', async () => {
    const frame = await renderFrame(
      baseInfo([
        mcp({ connected: true, name: 'xiaoban-support', status: 'connected', tools: 6 }),
        mcp({ connected: false, disabled: true, name: 'linear', status: 'disabled' })
      ])
    )

    expect(frame).toContain('Xiaoban')
    expect(frame).toContain('Welcome back!')
    expect(frame).toContain('My Stand')
    expect(frame).toContain('Tips for getting started')
    expect(frame).toContain('Recent activity')
    expect(frame).not.toContain('Available Tools')
    expect(frame).not.toContain('Available Skills')
    expect(frame).not.toContain('MCP servers')
    expect(frame).not.toMatch(/\d MCP\b/)
    expect(frame).not.toMatch(/\d tools\b/)
    expect(frame).not.toMatch(/\d skills\b/)
  })
})
