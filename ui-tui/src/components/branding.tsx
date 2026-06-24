import { Box, Text, useStdout } from '@xiaoban/ink'
import { useEffect, useState } from 'react'
import unicodeSpinners from 'unicode-animations'

import { artWidth, heroMark, HERO_MARK_WIDTH, logo, LOGO_WIDTH, STARTUP_LEFT_WIDTH, STARTUP_MARK_LABEL_START } from '../banner.js'
import { flat } from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { PanelSection, SessionInfo } from '../types.js'

const LOADER_TICK_MS = 120

function InlineLoader({ label, t }: { label: string; t: Theme }) {
  const [tick, setTick] = useState(0)
  const spinner = unicodeSpinners.braille
  const frame = spinner.frames[tick % spinner.frames.length] ?? '⠋'

  useEffect(() => {
    const id = setInterval(() => setTick(n => n + 1), Math.max(LOADER_TICK_MS, spinner.interval))

    return () => clearInterval(id)
  }, [spinner.interval])

  return (
    <Text color={t.color.muted} wrap="truncate">
      <Text color={t.color.accent}>{frame}</Text> {label}
    </Text>
  )
}

export function ArtLines({ lines, width }: { lines: [string, string][]; width?: number }) {
  const bodyW = artWidth(lines)
  const pad = width ? Math.max(0, (width - bodyW) >> 1) : 0

  return (
    <Box flexDirection="column" height={lines.length} opaque width={width ?? bodyW}>
      {lines.map(([c, text], i) => (
        <Text color={c} key={i} wrap="truncate-end">
          {' '.repeat(pad)}{text}
        </Text>
      ))}
    </Box>
  )
}

// Responsive Banner: full art → compact rule → text → hidden.
//
// Terminals can't scale glyphs, so "responsive" means picking a layout that
// fits the available columns. Thresholds are picked so each tier reads
// comfortably without forcing wrap or truncation drift on box-drawing edges.
const TAG_FULL = 'My Stand native agent'
const TAG_MID = 'Native agent'
const TAG_TINY = 'My Stand'
const HIDE_BELOW = 34
const COMPACT_FROM = 58

const clip = (s: string, w: number) =>
  w <= 0 ? '' : s.length > w ? `${s.slice(0, Math.max(0, w - 1))}…` : s

const centerIn = (s: string, w: number) => {
  const f = clip(s, w)
  const slack = Math.max(0, w - f.length)
  const left = slack >> 1

  return `${' '.repeat(left)}${f}${' '.repeat(slack - left)}`
}

type TextSegment = {
  bold?: boolean
  color: string
  text: string
}

const segmentLength = (segments: TextSegment[]) => segments.reduce((n, segment) => n + segment.text.length, 0)

const padSegments = (segments: TextSegment[], width: number, align: 'center' | 'left' = 'left') => {
  const len = segmentLength(segments)
  const slack = Math.max(0, width - len)
  const left = align === 'center' ? slack >> 1 : 0
  const right = slack - left
  const color = segments[0]?.color ?? 'white'

  return [
    ...(left > 0 ? [{ color, text: ' '.repeat(left) }] : []),
    ...segments,
    ...(right > 0 ? [{ color, text: ' '.repeat(right) }] : [])
  ]
}

const placeSegments = (segments: TextSegment[], width: number, start: number) => {
  const len = segmentLength(segments)
  const left = Math.max(0, Math.min(start, width - len))
  const right = Math.max(0, width - len - left)
  const color = segments[0]?.color ?? 'white'

  return [
    ...(left > 0 ? [{ color, text: ' '.repeat(left) }] : []),
    ...segments,
    ...(right > 0 ? [{ color, text: ' '.repeat(right) }] : [])
  ]
}

function SegmentText({ segments }: { segments: TextSegment[] }) {
  return (
    <>
      {segments.map((segment, i) => (
        <Text bold={segment.bold} color={segment.color} key={i}>
          {segment.text}
        </Text>
      ))}
    </>
  )
}

const ruleIn = (label: string, w: number) => {
  const f = clip(label, Math.max(1, w - 4))
  const slack = Math.max(0, w - f.length - 2)
  const left = slack >> 1

  return `${'─'.repeat(left)} ${f} ${'─'.repeat(slack - left)}`
}

function CompactBanner({ cols, t }: { cols: number; t: Theme }) {
  // -4 keeps a margin so exact-edge rows don't trip terminal pending-wrap.
  const w = Math.max(28, cols - 4)

  return (
    <Box flexDirection="column" height={3} marginBottom={1} opaque width={w}>
      <Text bold color={t.color.primary}>{ruleIn(t.brand.name, w)}</Text>
      <Text color={t.color.muted}>{centerIn(TAG_FULL, w)}</Text>
      <Text color={t.color.primary}>{'─'.repeat(w)}</Text>
    </Box>
  )
}

export function Banner({ maxWidth, t }: { maxWidth?: number; t: Theme }) {
  const term = useStdout().stdout?.columns ?? 80
  const cols = Math.max(1, Math.min(term, maxWidth ?? term))

  if (cols < HIDE_BELOW) {
    return null
  }

  const logoLines = logo(t.color, t.bannerLogo || undefined)
  const logoW = t.bannerLogo ? artWidth(logoLines) : LOGO_WIDTH

  if (cols >= logoW + 2) {
    return (
      <Box flexDirection="column" marginBottom={1}>
        <ArtLines lines={logoLines} />
        <Text color={t.color.muted} wrap="truncate-end">
          {t.brand.icon} {TAG_FULL}
        </Text>
      </Box>
    )
  }

  if (cols >= COMPACT_FROM) {
    return <CompactBanner cols={cols} t={t} />
  }

  const name = cols >= 52 ? t.brand.name : (t.brand.name.split(' ')[0] ?? t.brand.name)
  const tag = cols >= 64 ? TAG_FULL : cols >= 46 ? TAG_MID : TAG_TINY

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text bold color={t.color.primary} wrap="truncate-end">{t.brand.icon} {name}</Text>
      <Text color={t.color.muted} wrap="truncate-end">{t.brand.icon} {tag}</Text>
    </Box>
  )
}

// ── Collapsible helpers ──────────────────────────────────────────────

function CollapseToggle({
  count,
  open,
  suffix,
  t,
  title,
  onToggle
}: {
  count?: number
  open: boolean
  suffix?: string
  t: Theme
  title: string
  onToggle: () => void
}) {
  return (
    <Box onClick={onToggle}>
      <Text color={t.color.accent}>{open ? '▾ ' : '▸ '}</Text>
      <Text bold color={t.color.accent}>
        {title}
      </Text>
      {typeof count === 'number' ? (
        <Text color={t.color.muted}> ({count})</Text>
      ) : null}
      {suffix ? (
        <Text color={t.color.muted}> {suffix}</Text>
      ) : null}
    </Box>
  )
}

// ── SessionPanel ─────────────────────────────────────────────────────

const SKILLS_MAX = 8
const TOOLSETS_MAX = 8

export function SessionPanel({ info, maxWidth, sid, t }: SessionPanelProps) {
  const term = useStdout().stdout?.columns ?? 100
  const cols = Math.max(20, Math.min(term, maxWidth ?? term))
  const heroLines = heroMark(t.color, t.bannerHero || undefined)
  const targetLeftW = Math.max(STARTUP_LEFT_WIDTH, (artWidth(heroLines) || HERO_MARK_WIDTH) + 4)
  const leftW = Math.min(targetLeftW, Math.floor(cols * 0.42))
  const wide = cols >= 82 && leftW + 34 < cols
  const w = Math.max(20, wide ? cols - leftW - 14 : cols - 12)
  const lineBudget = Math.max(12, w - 2)
  const strip = (s: string) => (s.endsWith('_tools') ? s.slice(0, -6) : s)

  // ── Local collapse state for each section ──
  const [toolsOpen, setToolsOpen] = useState(false)
  const [skillsOpen, setSkillsOpen] = useState(false)
  const [systemOpen, setSystemOpen] = useState(false)
  const [mcpOpen, setMcpOpen] = useState(false)

  const truncLine = (pfx: string, items: string[]) => {
    let line = ''
    let shown = 0

    for (const item of [...items].sort()) {
      const next = line ? `${line}, ${item}` : item

      if (pfx.length + next.length > lineBudget) {
        return line ? `${line}, …+${items.length - shown}` : `${item}, …`
      }

      line = next
      shown++
    }

    return line
  }

  // ── Collapsible skills section ──
  const skillEntries = Object.entries(info.skills).sort()
  const skillsTotal = flat(info.skills).length
  const skillsCatCount = skillEntries.length

  const skillsBody = () => {
    if (info.lazy && skillEntries.length === 0) {
      return <InlineLoader label="scanning skills" t={t} />
    }

    const shown = skillEntries.slice(0, SKILLS_MAX)
    const overflow = skillEntries.length - SKILLS_MAX

    return (
      <>
        {shown.map(([k, vs]) => (
          <Text key={k} wrap="truncate">
            <Text color={t.color.muted}>{strip(k)}: </Text>
            <Text color={t.color.text}>{truncLine(strip(k) + ': ', vs)}</Text>
          </Text>
        ))}
        {overflow > 0 && (
          <Text color={t.color.muted}>(and {overflow} more categories…)</Text>
        )}
      </>
    )
  }

  // ── Collapsible tools section ──
  const toolEntries = Object.entries(info.tools).sort()
  const toolsTotal = flat(info.tools).length

  // MCP headline counts *connected* servers, not configured-but-disabled ones,
  // so it matches the classic CLI banner (`sum(s.connected)` in
  // xiaoban_cli/banner.py) and the "connected" label on the collapse toggle.
  const mcpServers = info.mcp_servers ?? []
  const mcpConnected = mcpServers.filter(s => s.connected).length

  const toolsBody = () => {
    const shown = toolEntries.slice(0, TOOLSETS_MAX)
    const overflow = toolEntries.length - TOOLSETS_MAX

    return (
      <>
        {shown.map(([k, vs]) => (
          <Text key={k} wrap="truncate">
            <Text color={t.color.muted}>{strip(k)}: </Text>
            <Text color={t.color.text}>{truncLine(strip(k) + ': ', vs)}</Text>
          </Text>
        ))}
        {overflow > 0 && (
          <Text color={t.color.muted}>(and {overflow} more toolsets…)</Text>
        )}
      </>
    )
  }

  // ── Collapsible MCP section ──
  const mcpBody = () => (
    <>
      {(info.mcp_servers ?? []).map(s => (
        <Text key={s.name} wrap="truncate">
          <Text color={t.color.muted}>{`  ${s.name} `}</Text>
          <Text color={t.color.muted}>{`[${s.transport}]`}</Text>
          <Text color={t.color.muted}>: </Text>
          {s.connected ? (
            <Text color={t.color.text}>
              {s.tools} tool{s.tools === 1 ? '' : 's'}
            </Text>
          ) : s.disabled || s.status === 'disabled' ? (
            <Text color={t.color.muted}>disabled</Text>
          ) : s.status === 'connecting' ? (
            <Text color={t.color.warn}>connecting</Text>
          ) : s.status === 'configured' ? (
            <Text color={t.color.muted}>configured</Text>
          ) : (
            <Text color={t.color.error}>failed</Text>
          )}
        </Text>
      ))}
    </>
  )

  // ── System prompt body ──
  const sysPromptLen = (info.system_prompt ?? '').length

  const systemBody = () => {
    if (sysPromptLen === 0) {
      return <Text color={t.color.muted}>No system prompt loaded.</Text>
    }

    return (
      <Text color={t.color.muted}>
        {info.system_prompt}
      </Text>
    )
  }

  const modelName = info.model.split('/').pop() ?? info.model
  const billingLabel = ' · API Usage Billing'
  const modelLabel = `${modelName}${billingLabel}`
  const modelPad = ' '.repeat(Math.max(0, (leftW - modelLabel.length) >> 1))
  const title = `${t.brand.name}${info.version ? ` v${info.version}` : ''}`

  if (wide) {
    const frameW = Math.max(82, cols)
    const leftCellW = STARTUP_LEFT_WIDTH + 4
    const rightCellW = Math.max(20, frameW - leftCellW - 3)
    const topRule = Math.max(0, frameW - title.length - 5)
    const markRows = heroLines.map(([color, text], i) => ({
      left: i === 0
        ? placeSegments([{ color, text: text.trimEnd() }], leftCellW, STARTUP_MARK_LABEL_START)
        : padSegments([{ color, text: text.trimEnd() }], leftCellW, 'center')
    }))
    const leftRows = [
      { left: padSegments([{ color: t.color.text, text: '' }], leftCellW) },
      { left: padSegments([{ color: t.color.text, text: 'Welcome back!' }], leftCellW, 'center') },
      { left: padSegments([{ color: t.color.text, text: '' }], leftCellW) },
      ...markRows,
      { left: padSegments([{ color: t.color.text, text: '' }], leftCellW) },
      {
        left: padSegments([
          { color: t.color.accent, text: modelName },
          { color: t.color.muted, text: billingLabel }
        ], leftCellW, 'center')
      },
      { left: padSegments([{ color: t.color.muted, text: info.cwd || process.cwd() }], leftCellW, 'center') }
    ]
    const rightRows = [
      [{ bold: true, color: t.color.primary, text: '  Tips for getting started' }],
      [{ color: t.color.text, text: '  Run /help to see Xiaoban commands' }],
      [{ color: t.color.text, text: '  Run /new to start a clean session' }],
      [{ color: t.color.border, text: `  ${'─'.repeat(Math.min(64, Math.max(12, rightCellW - 4)))}` }],
      [{ bold: true, color: t.color.primary, text: '  Recent activity' }],
      [{ color: t.color.muted, text: '  No recent activity' }]
    ]

    return (
      <Box flexDirection="column" marginBottom={1} width={frameW}>
        <Text wrap="truncate-end">
          <Text color={t.color.border}>╭─ </Text>
          <Text bold color={t.color.primary}>{title}</Text>
          <Text color={t.color.border}> {'─'.repeat(topRule)}╮</Text>
        </Text>

        {leftRows.map((row, i) => (
          <Text key={i} wrap="truncate-end">
            <Text color={t.color.border}>│</Text>
            <SegmentText segments={row.left} />
            <Text color={t.color.border}>│</Text>
            <SegmentText segments={padSegments(rightRows[i] ?? [{ color: t.color.text, text: '' }], rightCellW)} />
            <Text color={t.color.border}>│</Text>
          </Text>
        ))}

        <Text color={t.color.border} wrap="truncate-end">╰{'─'.repeat(frameW - 2)}╯</Text>
      </Box>
    )
  }

  return (
    <Box borderColor={t.color.border} borderStyle="round" flexDirection="column" marginBottom={1} paddingX={2} paddingY={1}>
      <Text bold color={t.color.primary} wrap="truncate-end">
        {title}
      </Text>

      <Box marginTop={1}>
        {wide && (
          <>
            <Box flexDirection="column" marginRight={2} width={leftW}>
              <Text color={t.color.text}>{centerIn('Welcome back!', leftW)}</Text>
              <Text />
              <ArtLines lines={heroLines} width={leftW} />
              <Text />
              <Text color={t.color.accent} wrap="truncate-end">
                {modelPad}{modelName}
                <Text color={t.color.muted}>{billingLabel}</Text>
              </Text>
              <Text color={t.color.muted} wrap="truncate-end">
                {centerIn(info.cwd || process.cwd(), leftW)}
              </Text>
            </Box>

            <Box flexDirection="column" marginRight={2}>
              {Array.from({ length: 9 }).map((_, i) => (
                <Text color={t.color.border} key={i}>│</Text>
              ))}
            </Box>
          </>
        )}

        <Box flexDirection="column" width={w}>
          {!wide && (
            <>
              <Text color={t.color.text}>Welcome back!</Text>
              <ArtLines lines={heroLines} />
              <Text color={t.color.accent} wrap="truncate-end">
                {modelName}
                <Text color={t.color.muted}> · API Usage Billing</Text>
              </Text>
              <Text color={t.color.muted} wrap="truncate-end">
                {info.cwd || process.cwd()}
              </Text>
              <Text />
            </>
          )}

          <Text bold color={t.color.primary}>Tips for getting started</Text>
          <Text color={t.color.text} wrap="truncate-end">Run /help to see Xiaoban commands</Text>
          <Text color={t.color.text} wrap="truncate-end">Run /new to start a clean session</Text>
          <Text color={t.color.border}>{'─'.repeat(Math.min(64, Math.max(12, w - 2)))}</Text>
          <Text bold color={t.color.primary}>Recent activity</Text>
          <Text color={t.color.muted}>No recent activity</Text>

        {typeof info.update_behind === 'number' && info.update_behind > 0 && (
          <Text bold color={t.color.warn}>
            ! {info.update_behind} {info.update_behind === 1 ? 'commit' : 'commits'} behind
            <Text bold={false} color={t.color.warn} dimColor>
              {' '}
              - run{' '}
            </Text>
            <Text bold color={t.color.warn}>
              {info.update_command || 'xiaoban update'}
            </Text>
            <Text bold={false} color={t.color.warn} dimColor>
              {' '}
              to update
            </Text>
          </Text>
        )}
      </Box>
    </Box>
    </Box>
  )
}

export function Panel({ sections, t, title }: PanelProps) {
  return (
    <Box borderColor={t.color.border} borderStyle="round" flexDirection="column" paddingX={2} paddingY={1}>
      <Box justifyContent="center" marginBottom={1}>
        <Text bold color={t.color.primary}>
          {title}
        </Text>
      </Box>

      {sections.map((sec, si) => (
        <Box flexDirection="column" key={si} marginTop={si > 0 ? 1 : 0}>
          {sec.title && (
            <Text bold color={t.color.accent}>
              {sec.title}
            </Text>
          )}

          {sec.rows?.map(([k, v], ri) => (
            <Text key={ri} wrap="truncate">
              <Text color={t.color.muted}>{k.padEnd(20)}</Text>
              <Text color={t.color.text}>{v}</Text>
            </Text>
          ))}

          {sec.items?.map((item, ii) => (
            <Text color={t.color.text} key={ii} wrap="truncate">
              {item}
            </Text>
          ))}

          {sec.text && <Text color={t.color.muted}>{sec.text}</Text>}
        </Box>
      ))}
    </Box>
  )
}

interface PanelProps {
  sections: PanelSection[]
  t: Theme
  title: string
}

interface SessionPanelProps {
  info: SessionInfo
  maxWidth?: number
  sid?: string | null
  t: Theme
}
