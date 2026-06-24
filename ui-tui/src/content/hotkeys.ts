import { isMac, isRemoteShell } from '../lib/platform.js'

const action = isMac ? 'Cmd' : 'Ctrl'
const paste = isMac ? 'Cmd' : 'Alt'

const copyHotkeys: [string, string][] = isMac
  ? [
      ['Cmd+C', '复制选中内容'],
      ['Ctrl+C', '中断 / 清空草稿 / 退出']
    ]
  : isRemoteShell()
    ? [
        ['Cmd+C', '终端转发时复制选中内容'],
        ['Ctrl+C', '复制选中内容 / 中断 / 清空草稿 / 退出']
      ]
    : [['Ctrl+C', '复制选中内容 / 中断 / 清空草稿 / 退出']]

export const HOTKEYS: [string, string][] = [
  ...copyHotkeys,
  [action + '+D', '退出'],
  [action + '+G / Alt+G', '打开 $EDITOR 编写长消息（Alt+G 兼容 VSCode/Cursor）'],
  [action + '+L', '重绘界面'],
  [paste + '+V / /paste', '粘贴文本；/paste 会附加剪贴板图片'],
  ['Tab', '使用当前补全项'],
  ['↑/↓', '选择补全 / 编辑队列 / 浏览历史'],
  ['Ctrl+X', '打开实时会话切换器（编辑队列消息时会删除该队列项）'],
  [action + '+A/E', '跳到行首 / 行尾'],
  [action + '+Z / ' + action + '+Y', '撤销 / 重做输入编辑'],
  [action + '+W', '删除一个词'],
  [action + '+U/K', '删除到行首 / 行尾'],
  [action + '+←/→', '按词跳转'],
  ['Home/End', '跳到行首 / 行尾'],
  ['Shift+Enter / Alt+Enter', '插入换行'],
  ['\\+Enter', '多行续写备用方式'],
  ['!<cmd>', '运行 shell 命令，例如 !ls、!git status'],
  ['{!<cmd>}', '把 shell 输出插入到消息里，例如 "branch is {!git branch --show-current}"']
]
