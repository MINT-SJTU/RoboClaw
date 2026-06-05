interface CloudProviderPanelProps {
  statusKnown?: boolean
  connected: boolean
  ready: boolean
  managed: boolean
  canRebind: boolean
  canShowDebug: boolean
  showDebug: boolean
  readySkuCount?: number
  totalSkuCount?: number
  readyImageCount?: number
  totalImageCount?: number
  runtimeEndpoint?: string
  onRebind: () => void
  onRefresh: () => void
  onToggleDebug: () => void
}

export function CloudProviderPanel({
  statusKnown = true,
  connected,
  ready,
  managed,
  canRebind,
  canShowDebug,
  showDebug,
  readySkuCount,
  totalSkuCount,
  readyImageCount,
  totalImageCount,
  runtimeEndpoint,
  onRebind,
  onRefresh,
  onToggleDebug,
}: CloudProviderPanelProps) {
  const hasCatalogDetails = readySkuCount !== undefined || readyImageCount !== undefined
  const statusText = connected
    ? ready
      ? `${managed ? '托管算力池' : 'SSH 实例'}已连接`
      : '实例未就绪，需要重新连接'
    : statusKnown
      ? '后端服务未连接'
      : '正在检查云端实例'

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-bd/50 bg-bg px-3 py-2 text-xs text-tx3">
      <div className="min-w-0">
        <div className="text-xs font-semibold text-tx2">云端实例</div>
        <div className="mt-0.5 truncate">{statusText}</div>
        {runtimeEndpoint && (
          <div className="mt-0.5 truncate font-mono text-[11px] text-tx3" title={runtimeEndpoint}>
            {runtimeEndpoint}
          </div>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {canRebind && (
          <button
            type="button"
            onClick={onRebind}
            className="rounded-md border border-bd px-2.5 py-1.5 text-[11px] font-semibold text-tx2 hover:border-ac hover:text-ac transition-colors"
          >
            重新绑定
          </button>
        )}
        <button
          type="button"
          onClick={onRefresh}
          className="rounded-md border border-bd px-2.5 py-1.5 text-[11px] font-semibold text-tx2 hover:border-ac hover:text-ac transition-colors"
        >
          刷新
        </button>
      </div>
      {canShowDebug && (
        <div className="flex flex-wrap items-center gap-2">
          {showDebug && hasCatalogDetails && (
            <span className="rounded-md border border-bd/60 bg-sf px-2.5 py-1 text-[11px] font-semibold text-tx3">
              机型 {String(readySkuCount ?? '-')} / {String(totalSkuCount ?? '-')} · 镜像 {String(readyImageCount ?? '-')} / {String(totalImageCount ?? '-')}
            </span>
          )}
          <button
            type="button"
            onClick={onToggleDebug}
            className="px-2.5 py-1 rounded-md border border-bd text-[11px] font-semibold text-tx2 hover:border-ac hover:text-ac transition-colors"
          >
            {showDebug ? '收起开发调试' : '开发调试'}
          </button>
        </div>
      )}
    </div>
  )
}
