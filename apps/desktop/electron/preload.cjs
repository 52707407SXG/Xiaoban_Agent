const { contextBridge, ipcRenderer, webUtils } = require('electron')

contextBridge.exposeInMainWorld('xiaobanDesktop', {
  getConnection: profile => ipcRenderer.invoke('xiaoban:connection', profile),
  revalidateConnection: () => ipcRenderer.invoke('xiaoban:connection:revalidate'),
  touchBackend: profile => ipcRenderer.invoke('xiaoban:backend:touch', profile),
  getGatewayWsUrl: profile => ipcRenderer.invoke('xiaoban:gateway:ws-url', profile),
  openSessionWindow: (sessionId, opts) => ipcRenderer.invoke('xiaoban:window:openSession', sessionId, opts),
  openNewSessionWindow: () => ipcRenderer.invoke('xiaoban:window:openNewSession'),
  getBootProgress: () => ipcRenderer.invoke('xiaoban:boot-progress:get'),
  getConnectionConfig: profile => ipcRenderer.invoke('xiaoban:connection-config:get', profile),
  saveConnectionConfig: payload => ipcRenderer.invoke('xiaoban:connection-config:save', payload),
  applyConnectionConfig: payload => ipcRenderer.invoke('xiaoban:connection-config:apply', payload),
  testConnectionConfig: payload => ipcRenderer.invoke('xiaoban:connection-config:test', payload),
  probeConnectionConfig: remoteUrl => ipcRenderer.invoke('xiaoban:connection-config:probe', remoteUrl),
  oauthLoginConnectionConfig: remoteUrl => ipcRenderer.invoke('xiaoban:connection-config:oauth-login', remoteUrl),
  oauthLogoutConnectionConfig: remoteUrl => ipcRenderer.invoke('xiaoban:connection-config:oauth-logout', remoteUrl),
  profile: {
    get: () => ipcRenderer.invoke('xiaoban:profile:get'),
    set: name => ipcRenderer.invoke('xiaoban:profile:set', name)
  },
  api: request => ipcRenderer.invoke('xiaoban:api', request),
  notify: payload => ipcRenderer.invoke('xiaoban:notify', payload),
  requestMicrophoneAccess: () => ipcRenderer.invoke('xiaoban:requestMicrophoneAccess'),
  readFileDataUrl: filePath => ipcRenderer.invoke('xiaoban:readFileDataUrl', filePath),
  readFileText: filePath => ipcRenderer.invoke('xiaoban:readFileText', filePath),
  selectPaths: options => ipcRenderer.invoke('xiaoban:selectPaths', options),
  writeClipboard: text => ipcRenderer.invoke('xiaoban:writeClipboard', text),
  saveImageFromUrl: url => ipcRenderer.invoke('xiaoban:saveImageFromUrl', url),
  saveImageBuffer: (data, ext) => ipcRenderer.invoke('xiaoban:saveImageBuffer', { data, ext }),
  saveClipboardImage: () => ipcRenderer.invoke('xiaoban:saveClipboardImage'),
  getPathForFile: file => {
    try {
      return webUtils.getPathForFile(file) || ''
    } catch {
      return ''
    }
  },
  normalizePreviewTarget: (target, baseDir) => ipcRenderer.invoke('xiaoban:normalizePreviewTarget', target, baseDir),
  watchPreviewFile: url => ipcRenderer.invoke('xiaoban:watchPreviewFile', url),
  stopPreviewFileWatch: id => ipcRenderer.invoke('xiaoban:stopPreviewFileWatch', id),
  setTitleBarTheme: payload => ipcRenderer.send('xiaoban:titlebar-theme', payload),
  setNativeTheme: mode => ipcRenderer.send('xiaoban:native-theme', mode),
  setTranslucency: payload => ipcRenderer.send('xiaoban:translucency', payload),
  setPreviewShortcutActive: active => ipcRenderer.send('xiaoban:previewShortcutActive', Boolean(active)),
  openExternal: url => ipcRenderer.invoke('xiaoban:openExternal', url),
  openPreviewInBrowser: url => ipcRenderer.invoke('xiaoban:openPreviewInBrowser', url),
  fetchLinkTitle: url => ipcRenderer.invoke('xiaoban:fetchLinkTitle', url),
  sanitizeWorkspaceCwd: cwd => ipcRenderer.invoke('xiaoban:workspace:sanitize', cwd),
  settings: {
    getDefaultProjectDir: () => ipcRenderer.invoke('xiaoban:setting:defaultProjectDir:get'),
    setDefaultProjectDir: dir => ipcRenderer.invoke('xiaoban:setting:defaultProjectDir:set', dir),
    pickDefaultProjectDir: () => ipcRenderer.invoke('xiaoban:setting:defaultProjectDir:pick')
  },
  revealLogs: () => ipcRenderer.invoke('xiaoban:logs:reveal'),
  getRecentLogs: () => ipcRenderer.invoke('xiaoban:logs:recent'),
  readDir: dirPath => ipcRenderer.invoke('xiaoban:fs:readDir', dirPath),
  gitRoot: startPath => ipcRenderer.invoke('xiaoban:fs:gitRoot', startPath),
  worktrees: cwds => ipcRenderer.invoke('xiaoban:fs:worktrees', cwds),
  terminal: {
    dispose: id => ipcRenderer.invoke('xiaoban:terminal:dispose', id),
    resize: (id, size) => ipcRenderer.invoke('xiaoban:terminal:resize', id, size),
    start: options => ipcRenderer.invoke('xiaoban:terminal:start', options),
    write: (id, data) => ipcRenderer.invoke('xiaoban:terminal:write', id, data),
    onData: (id, callback) => {
      const channel = `xiaoban:terminal:${id}:data`
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on(channel, listener)
      return () => ipcRenderer.removeListener(channel, listener)
    },
    onExit: (id, callback) => {
      const channel = `xiaoban:terminal:${id}:exit`
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on(channel, listener)
      return () => ipcRenderer.removeListener(channel, listener)
    }
  },
  onClosePreviewRequested: callback => {
    const listener = () => callback()
    ipcRenderer.on('xiaoban:close-preview-requested', listener)
    return () => ipcRenderer.removeListener('xiaoban:close-preview-requested', listener)
  },
  onOpenUpdatesRequested: callback => {
    const listener = () => callback()
    ipcRenderer.on('xiaoban:open-updates', listener)
    return () => ipcRenderer.removeListener('xiaoban:open-updates', listener)
  },
  onDeepLink: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('xiaoban:deep-link', listener)
    return () => ipcRenderer.removeListener('xiaoban:deep-link', listener)
  },
  signalDeepLinkReady: () => ipcRenderer.invoke('xiaoban:deep-link-ready'),
  onWindowStateChanged: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('xiaoban:window-state-changed', listener)
    return () => ipcRenderer.removeListener('xiaoban:window-state-changed', listener)
  },
  onFocusSession: callback => {
    const listener = (_event, sessionId) => callback(sessionId)
    ipcRenderer.on('xiaoban:focus-session', listener)
    return () => ipcRenderer.removeListener('xiaoban:focus-session', listener)
  },
  onNotificationAction: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('xiaoban:notification-action', listener)
    return () => ipcRenderer.removeListener('xiaoban:notification-action', listener)
  },
  onPreviewFileChanged: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('xiaoban:preview-file-changed', listener)
    return () => ipcRenderer.removeListener('xiaoban:preview-file-changed', listener)
  },
  onBackendExit: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('xiaoban:backend-exit', listener)
    return () => ipcRenderer.removeListener('xiaoban:backend-exit', listener)
  },
  onPowerResume: callback => {
    const listener = () => callback()
    ipcRenderer.on('xiaoban:power-resume', listener)
    return () => ipcRenderer.removeListener('xiaoban:power-resume', listener)
  },
  onBootProgress: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('xiaoban:boot-progress', listener)
    return () => ipcRenderer.removeListener('xiaoban:boot-progress', listener)
  },
  // First-launch bootstrap progress -- emitted by the install.ps1 stage
  // runner in main.cjs (apps/desktop/electron/bootstrap-runner.cjs).
  // Renderer's install overlay subscribes to live events and queries the
  // current snapshot via getBootstrapState() to recover after a devtools
  // reload mid-bootstrap.
  getBootstrapState: () => ipcRenderer.invoke('xiaoban:bootstrap:get'),
  resetBootstrap: () => ipcRenderer.invoke('xiaoban:bootstrap:reset'),
  repairBootstrap: () => ipcRenderer.invoke('xiaoban:bootstrap:repair'),
  cancelBootstrap: () => ipcRenderer.invoke('xiaoban:bootstrap:cancel'),
  onBootstrapEvent: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('xiaoban:bootstrap:event', listener)
    return () => ipcRenderer.removeListener('xiaoban:bootstrap:event', listener)
  },
  getVersion: () => ipcRenderer.invoke('xiaoban:version'),
  getRemoteDisplayReason: () => ipcRenderer.invoke('xiaoban:get-remote-display-reason'),
  uninstall: {
    summary: () => ipcRenderer.invoke('xiaoban:uninstall:summary'),
    run: mode => ipcRenderer.invoke('xiaoban:uninstall:run', { mode })
  },
  updates: {
    check: () => ipcRenderer.invoke('xiaoban:updates:check'),
    apply: opts => ipcRenderer.invoke('xiaoban:updates:apply', opts),
    getBranch: () => ipcRenderer.invoke('xiaoban:updates:branch:get'),
    setBranch: name => ipcRenderer.invoke('xiaoban:updates:branch:set', name),
    onProgress: callback => {
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on('xiaoban:updates:progress', listener)
      return () => ipcRenderer.removeListener('xiaoban:updates:progress', listener)
    }
  },
  themes: {
    fetchMarketplace: id => ipcRenderer.invoke('xiaoban:vscode-theme:fetch', id),
    searchMarketplace: query => ipcRenderer.invoke('xiaoban:vscode-theme:search', query)
  }
})
