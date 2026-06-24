import { forceRedraw, type MouseTrackingMode } from '@xiaoban/ink'

import { DASHBOARD_TUI_MODE, NO_CONFIRM_DESTRUCTIVE } from '../../../config/env.js'
import { dailyFortune, randomFortune } from '../../../content/fortunes.js'
import { HOTKEYS } from '../../../content/hotkeys.js'
import { isSectionName, nextDetailsMode, parseDetailsMode, SECTION_NAMES } from '../../../domain/details.js'
import type {
  ConfigGetValueResponse,
  ConfigSetResponse,
  SessionSaveResponse,
  SessionStatusResponse,
  SessionSteerResponse,
  SessionTitleResponse,
  SessionUndoResponse
} from '../../../gatewayTypes.js'
import { writeClipboardText } from '../../../lib/clipboard.js'
import { writeOsc52Clipboard } from '../../../lib/osc52.js'
import { configureDetectedTerminalKeybindings, configureTerminalKeybindings } from '../../../lib/terminalSetup.js'
import type { Msg, PanelSection } from '../../../types.js'
import type { StatusBarMode } from '../../interfaces.js'
import { patchOverlayState } from '../../overlayStore.js'
import { patchUiState } from '../../uiStore.js'
import type { SlashCommand } from '../types.js'

const flagFromArg = (arg: string, current: boolean): boolean | null => {
  if (!arg) {
    return !current
  }

  const mode = arg.trim().toLowerCase()

  if (mode === 'on') {
    return true
  }

  if (mode === 'off') {
    return false
  }

  if (mode === 'toggle') {
    return !current
  }

  return null
}

// `/mouse` toggles between full tracking and off when called bare so the
// old binary muscle-memory still works. Explicit presets (wheel / buttons /
// all) target the tmux-friendly hover-free subsets.
const MOUSE_MODE_ALIASES: Record<string, MouseTrackingMode> = {
  all: 'all',
  any: 'all',
  button: 'buttons',
  buttons: 'buttons',
  click: 'buttons',
  full: 'all',
  off: 'off',
  on: 'all',
  scroll: 'wheel',
  wheel: 'wheel'
}

const mouseModeFromArg = (arg: string, current: MouseTrackingMode): MouseTrackingMode | null => {
  if (!arg || arg.trim().toLowerCase() === 'toggle') {
    return current === 'off' ? 'all' : 'off'
  }

  return MOUSE_MODE_ALIASES[arg.trim().toLowerCase()] ?? null
}

const RESET_WORDS = new Set(['reset', 'clear', 'default'])
const CYCLE_WORDS = new Set(['cycle', 'toggle'])

const DETAILS_USAGE =
  'usage: /details [hidden|collapsed|expanded|cycle]  or  /details <section> [hidden|collapsed|expanded|reset]'

const DETAILS_SECTION_USAGE = 'usage: /details <section> [hidden|collapsed|expanded|reset]'

// Shown when /exit or /quit is refused in the hosted dashboard chat. Kept as a
// constant so the test asserts against the same source of truth as production.
export const DASHBOARD_EXIT_DISABLED_MESSAGE =
  '托管网页聊天里不能直接退出；请用 /new 开启新会话'

export const DASHBOARD_UPDATE_DISABLED_MESSAGE =
  '托管网页聊天里不能直接更新；该环境由服务端统一管理'

export const coreCommands: SlashCommand[] = [
  {
    help: '查看命令和快捷键',
    name: 'help',
    run: (_arg, ctx) => {
      const sections: PanelSection[] = (ctx.local.catalog?.categories ?? []).map(cat => ({
        rows: cat.pairs,
        title: cat.name
      }))

      if (ctx.local.catalog?.skillCount) {
        sections.push({ text: `还有 ${ctx.local.catalog.skillCount} 个技能命令，可用 /skills 浏览` })
      }

      sections.push(
        {
          rows: [
            ['/details [hidden|collapsed|expanded|cycle]', '设置 Agent 细节展示模式'],
            [
              '/details <section> [hidden|collapsed|expanded|reset]',
              '单独设置某一块细节（thinking/tools/subagents/activity）'
            ],
            ['/fortune [random|daily]', '显示一条随机或每日小提示']
          ],
          title: 'TUI'
        },
        { rows: HOTKEYS, title: '快捷键' }
      )

      ctx.transcript.panel(ctx.ui.theme.brand.helpHeader, sections)
    }
  },

  {
    aliases: ['exit'],
    help: '退出小伴 CLI',
    name: 'quit',
    run: (_arg, ctx) => {
      // In the hosted dashboard chat there is no in-page restart path after
      // the PTY child exits, so quitting bricks the tab until a refresh. The
      // keyboard idle-exit (Ctrl+C / Ctrl+D) and SIGINT handling already refuse
      // to die in this mode (see useInputHandlers + entry.tsx); gate /exit and
      // /quit on the same DASHBOARD_TUI_MODE flag. Unlike the keyboard path
      // (which auto-starts a fresh chat), the explicit quit command refuses and
      // instructs the user to run /new themselves.
      if (DASHBOARD_TUI_MODE) {
        ctx.transcript.sys(DASHBOARD_EXIT_DISABLED_MESSAGE)

        return
      }

      ctx.session.die()
    }
  },

  {
    help: '更新小伴到最新版本（会退出当前 TUI）',
    name: 'update',
    run: (_arg, ctx) => {
      if (DASHBOARD_TUI_MODE) {
        ctx.transcript.sys(DASHBOARD_UPDATE_DISABLED_MESSAGE)

        return
      }

      ctx.transcript.sys('正在退出 TUI 并执行更新...')
      // Exit code 42 signals the Python wrapper to exec `xiaoban update`.
      // Use dieWithCode for proper cleanup (gateway kill + Ink unmount).
      setTimeout(() => ctx.session.dieWithCode(42), 100)
    }
  },

  {
    aliases: ['scroll'],
    help: '设置鼠标跟踪模式 [on|off|toggle|wheel|buttons|all]',
    name: 'mouse',
    run: (arg, ctx) => {
      const current = ctx.ui.mouseTracking
      const next = mouseModeFromArg(arg, current)

      if (next === null) {
        return ctx.transcript.sys('usage: /mouse [on|off|toggle|wheel|buttons|all]')
      }

      patchUiState({ mouseTracking: next })
      ctx.gateway.rpc<ConfigSetResponse>('config.set', { key: 'mouse', value: next }).catch(() => {})

      queueMicrotask(() => ctx.transcript.sys(`mouse tracking ${next}`))
    }
  },

  {
    aliases: ['new'],
    help: '开启新会话',
    name: 'clear',
    run: (arg, ctx, cmd) => {
      if (ctx.session.guardBusySessionSwitch('switch sessions')) {
        return
      }

      const isNew = cmd.startsWith('/new')
      const requestedTitle = isNew ? arg.trim() : ''

      const commit = () => {
        patchUiState({ status: 'forging session…' })
        ctx.session.newSession(isNew ? 'new session started' : undefined, requestedTitle || undefined)
      }

      if (NO_CONFIRM_DESTRUCTIVE) {
        return commit()
      }

      patchOverlayState({
        confirm: {
          cancelLabel: '不，继续当前会话',
          confirmLabel: isNew ? '确认开启新会话' : '确认清空当前会话',
          danger: true,
          detail: '这会结束当前对话，并清空屏幕里的聊天记录。',
          onConfirm: commit,
          title: isNew ? '开启新会话？' : '清空当前会话？'
        }
      })
    }
  },

  {
    help: '强制重绘界面',
    name: 'redraw',
    run: (_arg, ctx) => {
      forceRedraw(process.stdout)
      ctx.transcript.sys('界面已重绘')
    }
  },

  {
    help: '查看当前会话信息',
    name: 'status',
    run: (_arg, ctx) => {
      if (!ctx.sid) {
        return ctx.transcript.sys('当前没有活动会话')
      }

      ctx.gateway
        .rpc<SessionStatusResponse>('session.status', { session_id: ctx.sid })
        .then(ctx.guarded<SessionStatusResponse>(r => ctx.transcript.page(r.output || '(no status)', 'Status')))
        .catch(ctx.guardedErr)
    }
  },

  {
    help: '设置或查看当前会话标题',
    name: 'title',
    run: (arg, ctx) => {
      if (!ctx.sid) {
        return ctx.transcript.sys('当前没有活动会话')
      }

      const title = arg.trim()

      if (!arg) {
        ctx.gateway
          .rpc<SessionTitleResponse>('session.title', { session_id: ctx.sid })
          .then(
            ctx.guarded<SessionTitleResponse>(r => {
              const current = (r?.title ?? '').trim()
              ctx.transcript.sys(current ? `title: ${current}` : 'no title set')
            })
          )
          .catch(ctx.guardedErr)

        return
      }

      if (!title) {
        return ctx.transcript.sys('usage: /title <your session title>')
      }

      ctx.gateway
        .rpc<SessionTitleResponse>('session.title', { session_id: ctx.sid, title })
        .then(
          ctx.guarded<SessionTitleResponse>(r => {
            const next = (r?.title ?? title).trim()
            const suffix = r?.pending ? ' (queued while session initializes)' : ''
            ctx.transcript.sys(`session title set: ${next}${suffix}`)
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: '切换紧凑聊天记录模式',
    name: 'compact',
    run: (arg, ctx) => {
      const next = flagFromArg(arg, ctx.ui.compact)

      if (next === null) {
        return ctx.transcript.sys('usage: /compact [on|off|toggle]')
      }

      patchUiState({ compact: next })
      ctx.gateway.rpc<ConfigSetResponse>('config.set', { key: 'compact', value: next ? 'on' : 'off' }).catch(() => {})

      queueMicrotask(() => ctx.transcript.sys(`compact ${next ? 'on' : 'off'}`))
    }
  },

  {
    aliases: ['detail'],
    help: '控制 Agent 细节展示，可全局或分块设置',
    name: 'details',
    run: (arg, ctx) => {
      const { gateway, transcript, ui } = ctx

      if (!arg) {
        gateway
          .rpc<ConfigGetValueResponse>('config.get', { key: 'details_mode' })
          .then(r => {
            if (ctx.stale()) {
              return
            }

            const mode = parseDetailsMode(r?.value) ?? ui.detailsMode
            patchUiState({ detailsMode: mode, detailsModeCommandOverride: false })

            const overrides = SECTION_NAMES.filter(s => ui.sections[s])
              .map(s => `${s}=${ui.sections[s]}`)
              .join(' ')

            transcript.sys(`details: ${mode}${overrides ? `  (${overrides})` : ''}`)
          })
          .catch(() => !ctx.stale() && transcript.sys(`details: ${ui.detailsMode}`))

        return
      }

      const [first, second] = arg.trim().toLowerCase().split(/\s+/)

      if (second && isSectionName(first)) {
        const reset = RESET_WORDS.has(second)
        const mode = reset ? null : parseDetailsMode(second)

        if (!reset && !mode) {
          return transcript.sys(DETAILS_SECTION_USAGE)
        }

        const { [first]: _drop, ...rest } = ui.sections

        patchUiState({ sections: mode ? { ...rest, [first]: mode } : rest })
        gateway
          .rpc<ConfigSetResponse>('config.set', { key: `details_mode.${first}`, value: mode ?? '' })
          .catch(() => {})
        transcript.sys(`details ${first}: ${mode ?? 'reset'}`)

        return
      }

      const next = CYCLE_WORDS.has(first ?? '') ? nextDetailsMode(ui.detailsMode) : parseDetailsMode(first)

      if (!next) {
        return transcript.sys(DETAILS_USAGE)
      }

      const sections = Object.fromEntries(SECTION_NAMES.map(section => [section, next]))

      patchUiState({ detailsMode: next, detailsModeCommandOverride: true, sections })
      gateway.rpc<ConfigSetResponse>('config.set', { key: 'details_mode', value: next }).catch(() => {})
      transcript.sys(`details: ${next}`)
    }
  },

  {
    help: '显示本地小提示',
    name: 'fortune',
    run: (arg, ctx) => {
      const key = arg.trim().toLowerCase()

      if (!arg || key === 'random') {
        return ctx.transcript.sys(randomFortune())
      }

      if (['daily', 'stable', 'today'].includes(key)) {
        return ctx.transcript.sys(dailyFortune(ctx.sid))
      }

      ctx.transcript.sys('usage: /fortune [random|daily]')
    }
  },

  {
    help: '复制选中文字或小伴回复',
    name: 'copy',
    run: async (arg, ctx) => {
      const { sys } = ctx.transcript

      if (!arg && ctx.composer.hasSelection) {
        const text = await ctx.composer.selection.copySelection()

        if (text) {
          return sys(`copied ${text.length} characters`)
        } else {
          return sys(
            '剪贴板复制失败；可尝试设置 XIAOBAN_TUI_FORCE_OSC52=1 强制使用终端复制序列'
          )
        }
      }

      if (arg && Number.isNaN(parseInt(arg, 10))) {
        return sys('usage: /copy [number]')
      }

      const all = ctx.local.getHistoryItems().filter(m => m.role === 'assistant')
      const target = all[arg ? Math.min(parseInt(arg, 10), all.length) - 1 : all.length - 1]

      if (!target) {
        return sys('nothing to copy — start a conversation first')
      }

      void writeClipboardText(target.text)
        .then(nativeOk => {
          if (ctx.stale()) {
            return
          }

          if (nativeOk) {
            sys('copied to clipboard')
          } else {
            writeOsc52Clipboard(target.text)
            sys('sent OSC52 copy sequence (terminal support required)')
          }
        })
        .catch(error => {
          if (!ctx.stale()) {
            sys(`copy failed: ${String(error)}`)
          }
        })
    }
  },

  {
    help: '附加剪贴板里的图片',
    name: 'paste',
    run: (arg, ctx) => (arg ? ctx.transcript.sys('usage: /paste') : ctx.composer.paste())
  },

  {
    aliases: ['compose'],
    help: '用 $EDITOR 编写下一条消息（同 Ctrl+G）',
    name: 'prompt',
    run: (arg, ctx) => {
      if (arg) {
        // The TUI editor opens with the current composer draft; there is no
        // separate seed arg. Drop any inline text into the composer first so
        // it carries into the editor, matching the CLI's /prompt <text>.
        ctx.composer.setInput(arg)
      }

      void ctx.composer.openEditor().catch((err: unknown) => {
        ctx.transcript.sys(`editor failed: ${String(err)}`)
      })
    }
  },

  {
    help: '配置 IDE 终端快捷键，支持多行输入和撤销重做',
    name: 'terminal-setup',
    run: (arg, ctx) => {
      const target = arg.trim().toLowerCase()

      if (target && !['auto', 'cursor', 'vscode', 'windsurf'].includes(target)) {
        return ctx.transcript.sys('usage: /terminal-setup [auto|vscode|cursor|windsurf]')
      }

      const runner =
        !target || target === 'auto'
          ? configureDetectedTerminalKeybindings()
          : configureTerminalKeybindings(target as 'cursor' | 'vscode' | 'windsurf')

      void runner
        .then(result => {
          if (ctx.stale()) {
            return
          }

          ctx.transcript.sys(result.message)

          if (result.success && result.requiresRestart) {
            ctx.transcript.sys('restart the IDE terminal for the new keybindings to take effect')
          }
        })
        .catch(error => {
          if (!ctx.stale()) {
            ctx.transcript.sys(`terminal setup failed: ${String(error)}`)
          }
        })
    }
  },

  {
    help: '查看网关日志',
    name: 'logs',
    run: (arg, ctx) => {
      const text = ctx.gateway.gw.getLogTail(Math.min(80, Math.max(1, parseInt(arg, 10) || 20)))

      text ? ctx.transcript.page(text, 'Logs') : ctx.transcript.sys('no gateway logs')
    }
  },

  {
    help: '查看当前聊天记录（用户和小伴消息）',
    name: 'history',
    run: (arg, ctx) => {
      // The CLI-side `/history` runs in a detached slash-worker subprocess
      // that never sees the TUI's turns — it only surfaces whatever was
      // persisted before this process started.  Render the TUI's own
      // transcript so `/history` actually reflects what the user just did.
      const items = ctx.local.getHistoryItems().filter(m => m.role === 'user' || m.role === 'assistant')

      if (!items.length) {
        return ctx.transcript.sys('no conversation yet')
      }

      const preview = Math.max(80, parseInt(arg, 10) || 400)

      const lines = items.map((m, i) => {
        const tag = m.role === 'user' ? `你 #${i + 1}` : `小伴 #${i + 1}`
        const body = m.text.trim() || (m.tools?.length ? `(${m.tools.length} tool calls)` : '(empty)')
        const clipped = body.length > preview ? `${body.slice(0, preview).trimEnd()}…` : body

        return `[${tag}]\n${clipped}`
      })

      ctx.transcript.page(lines.join('\n\n'), 'History')
    }
  },

  {
    help: '把当前聊天记录保存成 JSON',
    name: 'save',
    run: (_arg, ctx) => {
      const hasConversation = ctx.local
        .getHistoryItems()
        .some(m => m.role === 'user' || m.role === 'assistant' || m.role === 'tool')

      if (!hasConversation) {
        return ctx.transcript.sys('no conversation yet')
      }

      if (!ctx.sid) {
        return ctx.transcript.sys('no active session — nothing to save')
      }

      ctx.gateway
        .rpc<SessionSaveResponse>('session.save', { session_id: ctx.sid })
        .then(
          ctx.guarded<SessionSaveResponse>(r => {
            const file = r?.file

            if (file) {
              ctx.transcript.sys(`conversation saved to: ${file}`)
            } else {
              ctx.transcript.sys('failed to save')
            }
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    aliases: ['sb'],
    help: '设置状态栏位置 [on|off|top|bottom]',
    name: 'statusbar',
    run: (arg, ctx) => {
      const mode = arg.trim().toLowerCase()
      const toggle: StatusBarMode = ctx.ui.statusBar === 'off' ? 'top' : 'off'

      const next: null | StatusBarMode =
        !mode || mode === 'toggle'
          ? toggle
          : mode === 'on' || mode === 'top'
            ? 'top'
            : mode === 'off' || mode === 'bottom'
              ? mode
              : null

      if (!next) {
        return ctx.transcript.sys('usage: /statusbar [on|off|top|bottom|toggle]')
      }

      patchUiState({ statusBar: next })
      ctx.gateway.rpc<ConfigSetResponse>('config.set', { key: 'statusbar', value: next }).catch(() => {})

      queueMicrotask(() => ctx.transcript.sys(`status bar ${next}`))
    }
  },

  {
    aliases: ['q'],
    help: '查看队列或把消息加入队列',
    name: 'queue',
    run: (arg, ctx) => {
      if (!arg) {
        return ctx.transcript.sys(`${ctx.composer.queueRef.current.length} queued message(s)`)
      }

      ctx.composer.enqueue(arg)
      ctx.transcript.sys(`queued: "${arg.slice(0, 50)}${arg.length > 50 ? '…' : ''}"`)
    }
  },

  {
    help: '在下一次工具调用后插入消息，不中断当前任务',
    name: 'steer',
    run: (arg, ctx) => {
      const payload = arg?.trim() ?? ''

      if (!payload) {
        return ctx.transcript.sys('usage: /steer <prompt>')
      }

      // If the agent isn't running, fall back to the queue so the user's
      // message isn't lost — identical semantics to the gateway handler.
      if (!ctx.ui.busy || !ctx.sid) {
        ctx.composer.enqueue(payload)
        ctx.transcript.sys(
          `no active turn — queued for next: "${payload.slice(0, 50)}${payload.length > 50 ? '…' : ''}"`
        )

        return
      }

      ctx.gateway
        .rpc<SessionSteerResponse>('session.steer', { session_id: ctx.sid, text: payload })
        .then(
          ctx.guarded<SessionSteerResponse>(r => {
            if (r?.status === 'queued') {
              ctx.transcript.sys(
                `steer queued — arrives after next tool call: "${payload.slice(0, 50)}${payload.length > 50 ? '…' : ''}"`
              )
            } else {
              ctx.transcript.sys('steer rejected')
            }
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: '撤销上一轮对话',
    name: 'undo',
    run: (_arg, ctx) => {
      if (!ctx.sid) {
        return ctx.transcript.sys('nothing to undo')
      }

      ctx.gateway.rpc<SessionUndoResponse>('session.undo', { session_id: ctx.sid }).then(
        ctx.guarded<SessionUndoResponse>(r => {
          if ((r.removed ?? 0) > 0) {
            ctx.transcript.setHistoryItems((prev: Msg[]) => ctx.transcript.trimLastExchange(prev))
            ctx.transcript.sys(`undid ${r.removed} messages`)
          } else {
            ctx.transcript.sys('nothing to undo')
          }
        })
      )
    }
  },

  {
    help: '重试上一条用户消息',
    name: 'retry',
    run: (_arg, ctx) => {
      const last = ctx.local.getLastUserMsg()

      if (!last) {
        return ctx.transcript.sys('nothing to retry')
      }

      if (!ctx.sid) {
        return ctx.transcript.send(last)
      }

      ctx.gateway.rpc<SessionUndoResponse>('session.undo', { session_id: ctx.sid }).then(
        ctx.guarded<SessionUndoResponse>(r => {
          if ((r.removed ?? 0) <= 0) {
            return ctx.transcript.sys('nothing to retry')
          }

          ctx.transcript.setHistoryItems((prev: Msg[]) => ctx.transcript.trimLastExchange(prev))
          ctx.transcript.send(last)
        })
      )
    }
  }
]
