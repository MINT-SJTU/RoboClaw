import type { ReactNode, RefObject } from 'react'

interface CloudQueuedTask {
  id: string
  text: string
  createdAt: string
}

interface CloudIntentPanelProps {
  intent: string
  queue: CloudQueuedTask[]
  helperText: string
  intentRef?: RefObject<HTMLTextAreaElement>
  sourcePanel: ReactNode
  actions: ReactNode
  onIntentChange: (value: string) => void
  onRemoveQueuedTask: (id: string) => void
}

export function CloudIntentPanel({
  intent,
  queue,
  helperText,
  intentRef,
  sourcePanel,
  actions,
  onIntentChange,
  onRemoveQueuedTask,
}: CloudIntentPanelProps) {
  return (
    <div className="mt-4 rounded-2xl border border-bd/70 bg-sf p-3">
      <textarea
        ref={intentRef}
        value={intent}
        onChange={(event) => onIntentChange(event.target.value)}
        rows={3}
        placeholder="描述你的训练目标，例如：用 LIBERO 数据集微调 pi0，跑 500 步看看效果"
        className="w-full bg-bg border border-bd text-tx px-3 py-2 rounded-xl text-sm leading-relaxed focus:outline-none focus:border-ac resize-none"
      />
      {queue.length > 0 && (
        <div className="mt-2 rounded-xl border border-bd/60 bg-bg px-3 py-2">
          <div className="flex items-center justify-between gap-3">
            <div className="text-xs font-semibold text-tx2">待处理队列 · {queue.length}</div>
            <div className="text-[11px] text-tx3">当前任务结束后依次处理</div>
          </div>
          <div className="mt-2 grid gap-1.5">
            {queue.slice(0, 3).map((item, index) => (
              <div key={item.id} className="grid grid-cols-[auto_1fr_auto] items-center gap-2 rounded-lg bg-sf px-2.5 py-1.5">
                <span className="grid h-5 w-5 place-items-center rounded-full bg-ac/10 text-[11px] font-semibold text-ac">
                  {index + 1}
                </span>
                <div className="truncate text-xs text-tx2" title={item.text}>{item.text}</div>
                <button
                  type="button"
                  onClick={() => onRemoveQueuedTask(item.id)}
                  className="rounded px-2 py-1 text-[11px] font-semibold text-tx3 hover:bg-bg hover:text-rd transition-colors"
                >
                  移除
                </button>
              </div>
            ))}
            {queue.length > 3 && (
              <div className="px-2 text-[11px] text-tx3">还有 {queue.length - 3} 条等待中</div>
            )}
          </div>
        </div>
      )}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
        {sourcePanel}
        <div className="flex flex-wrap items-center gap-2">
          {actions}
        </div>
      </div>
      <div className="mt-2 text-xs text-tx3">{helperText}</div>
    </div>
  )
}
