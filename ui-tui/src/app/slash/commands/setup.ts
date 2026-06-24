import { withInkSuspended } from '@xiaoban/ink'

import { launchXiaobanCommand } from '../../../lib/externalCli.js'
import { runExternalSetup } from '../../setupHandoff.js'
import type { SlashCommand } from '../types.js'

export const setupCommands: SlashCommand[] = [
  {
    help: '运行完整设置向导（会启动 `xiaoban setup`）',
    name: 'setup',
    run: (arg, ctx) =>
      void runExternalSetup({
        args: ['setup', ...arg.split(/\s+/).filter(Boolean)],
        ctx,
        done: 'setup complete — starting session…',
        launcher: launchXiaobanCommand,
        suspend: withInkSuspended
      })
  }
]
