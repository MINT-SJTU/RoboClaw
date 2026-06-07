import type { RefObject } from 'react'

interface CloudSourcePanelProps {
  open: boolean
  sourceUri: string
  modelUri: string
  sourceInputRef?: RefObject<HTMLInputElement>
  modelInputRef?: RefObject<HTMLInputElement>
  onToggle: (open: boolean) => void
  onSourceUriChange: (value: string) => void
  onModelUriChange: (value: string) => void
}

export function CloudSourcePanel({
  open,
  sourceUri,
  modelUri,
  sourceInputRef,
  modelInputRef,
  onToggle,
  onSourceUriChange,
  onModelUriChange,
}: CloudSourcePanelProps) {
  return (
    <details
      open={open}
      onToggle={(event) => onToggle(event.currentTarget.open)}
      className="text-xs text-tx3"
    >
      <summary className="cursor-pointer select-none rounded-md border border-bd px-3 py-1.5 font-semibold text-tx2 hover:border-ac hover:text-ac">
        粘贴数据集或模型链接（可选）
      </summary>
      <div className="mt-3 grid gap-3 rounded-md border border-bd/50 bg-bg px-3 py-2">
        <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
          数据集链接
          <input
            ref={sourceInputRef}
            value={sourceUri}
            onChange={(event) => onSourceUriChange(event.target.value)}
            placeholder="HuggingFace / ModelScope 地址，留空由 AI 自动匹配"
            className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
          />
        </label>
        <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
          模型链接
          <input
            ref={modelInputRef}
            value={modelUri}
            onChange={(event) => onModelUriChange(event.target.value)}
            placeholder="模型仓库或 checkpoint 地址，留空则由总控判断"
            className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
          />
        </label>
        <div className="text-[11px] text-tx3 leading-relaxed">
          只准备、直接运行、从头训练这些都可以直接写在上方需求里。
        </div>
      </div>
    </details>
  )
}
