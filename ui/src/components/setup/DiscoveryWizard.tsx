import { useState, useEffect } from 'react'
import { useSetup, deviceLabel } from '../../controllers/setup'
import { useI18n } from '../../controllers/i18n'
import type { CatalogModel } from '../../controllers/setup'
import ScanArea from './ScanArea'
import DeviceNode from './DeviceNode'

const STEPS = ['select', 'scan', 'identify', 'review'] as const
const btnBack = 'px-4 py-1.5 text-sm text-tx2 hover:text-tx transition-colors'
const btnPrimary = 'px-4 py-1.5 text-sm bg-ac text-white rounded-lg font-medium hover:bg-ac2 disabled:opacity-40 transition-colors'
const btnOutline = 'px-4 py-1.5 text-sm border border-ac/50 text-ac rounded-lg hover:border-ac hover:bg-ac/5 transition-colors'
const formBox = 'mt-1.5 ml-11 p-3 bg-sf border border-bd/30 rounded-lg shadow-card space-y-2'
const inputCls = 'w-full px-3 py-1.5 text-sm bg-sf2 border border-bd text-tx rounded focus:border-ac focus:shadow-glow-ac outline-none'

// -- Step Indicator ----------------------------------------------------------

function StepIndicator({ current }: { current: string }) {
  const idx = STEPS.indexOf(current as (typeof STEPS)[number])
  const labels = ['选择型号', '扫描', '分配', '确认']

  return (
    <div className="flex items-center justify-center gap-1 mb-6">
      {STEPS.map((s, i) => (
        <div key={s} className="flex items-center gap-1">
          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-2xs font-medium transition-colors ${
            i === idx ? 'bg-ac/10 text-ac' : i < idx ? 'text-gn' : 'text-tx2'
          }`}>
            <span className={`w-5 h-5 flex items-center justify-center rounded-full text-2xs ${
              i < idx ? 'bg-gn text-white' : i === idx ? 'bg-ac text-white' : 'bg-sf2 text-tx3'
            }`}>
              {i < idx ? '✓' : i + 1}
            </span>
            <span className="hidden sm:inline">{labels[i]}</span>
          </div>
          {i < STEPS.length - 1 && (
            <div className={`w-6 h-px ${i < idx ? 'bg-gn/60' : 'bg-bd'}`} />
          )}
        </div>
      ))}
    </div>
  )
}

// -- Step 1: Model Select ----------------------------------------------------

function ModelSelect() {
  const { catalog, selectedCategory, selectedModel, setCategory, setModel, goToStep, cancelWizard } =
    useSetup()
  const categories = catalog?.categories ?? []
  const models: CatalogModel[] =
    selectedCategory && catalog?.models ? catalog.models[selectedCategory] ?? [] : []

  const categoryIcons: Record<string, string> = { arm: '🦾', hand: '🤚', humanoid: '🤖', mobile: '🚗' }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {categories.map((cat) => (
          <button key={cat.id} disabled={!cat.supported} onClick={() => setCategory(cat.id)}
            className={`flex flex-col items-center gap-1 px-4 py-3 rounded-lg border text-sm font-medium transition-all ${
              !cat.supported ? 'border-bd/20 bg-sf text-tx3 opacity-50 cursor-not-allowed'
              : cat.id === selectedCategory ? 'border-ac bg-ac/5 text-ac shadow-card ring-1 ring-ac/20' : 'border-bd/30 bg-white hover:border-ac/40 shadow-card text-tx'
            }`}>
            <span className="text-lg">{categoryIcons[cat.id] || '📦'}</span>
            <span>{cat.id}</span>
            {!cat.supported && <span className="text-2xs text-tx2/60">即将支持</span>}
          </button>
        ))}
      </div>
      {selectedCategory && models.length > 0 && (
        <div className="flex gap-2 flex-wrap">
          {models.map((m) => (
            <button key={m.name} onClick={() => { setModel(m.name); goToStep('scan') }}
              className={`px-4 py-2 rounded-lg border text-sm font-medium transition-all ${
                m.name === selectedModel ? 'border-ac bg-ac/5 text-ac shadow-card ring-1 ring-ac/20' : 'border-bd/30 bg-white hover:border-ac/40 shadow-card text-tx'
              }`}>
              {m.name}
              {m.roles.length > 0 && <span className="text-2xs text-tx2 ml-1.5">({m.roles.join('/')})</span>}
            </button>
          ))}
        </div>
      )}
      <div className="flex justify-end pt-2">
        <button onClick={cancelWizard} className={btnBack}>取消</button>
      </div>
    </div>
  )
}

// -- Step 2: Scan ------------------------------------------------------------

function ScanStep() {
  const { scannedPorts, scannedCameras, scanning, doScan, goToStep } = useSetup()
  useEffect(() => { doScan() }, [])

  const hasDevices = scannedPorts.length > 0 || scannedCameras.length > 0

  return (
    <div className="space-y-4">
      <ScanArea ports={scannedPorts} cameras={scannedCameras} scanning={scanning} />
      <div className="flex justify-between pt-2">
        <button onClick={() => goToStep('select')} className={btnBack}>返回</button>
        <div className="flex gap-2">
          {!scanning && (
            <button onClick={doScan} className={btnOutline}>重新扫描</button>
          )}
          {!scanning && hasDevices && (
            <button onClick={() => goToStep('identify')} className={btnPrimary}>下一步</button>
          )}
        </div>
      </div>
    </div>
  )
}

// -- Step 3: Identify --------------------------------------------------------

function PortAssignForm({ stableId, roles, model }: {
  stableId: string; roles: string[]; model: string
}) {
  const { sessionAssign } = useSetup()
  const [alias, setAlias] = useState('')
  const [role, setRole] = useState('')

  function submit() {
    if (!alias.trim()) return
    const combined = role ? `${alias.trim()}_${role}` : alias.trim()
    const spec = role ? `${model}_${role}` : model
    sessionAssign(stableId, combined, spec)
  }

  return (
    <div className={formBox}>
      {roles.length > 0 && (
        <div className="flex gap-1.5 flex-wrap">
          {roles.map((r) => (
            <button key={r} onClick={() => setRole(r)}
              className={`px-2.5 py-1 text-2xs rounded-md border transition-colors ${
                role === r ? 'border-ac bg-ac/5 text-ac font-medium ring-1 ring-ac/20' : 'border-bd/50 text-tx2 hover:border-ac/50'
              }`}>
              {r}
            </button>
          ))}
        </div>
      )}
      <input value={alias} onChange={(e) => setAlias(e.target.value)}
        placeholder="设备名称" className={inputCls} />
      {alias.trim() && (
        <p className="text-2xs text-tx2">预览: {role ? `${alias.trim()}_${role}` : alias.trim()}</p>
      )}
      <button onClick={submit} disabled={!alias.trim() || (roles.length > 0 && !role)}
        className="px-3 py-1 text-xs bg-ac text-white rounded-lg font-medium disabled:opacity-40 hover:bg-ac2 transition-colors">
        确认分配
      </button>
    </div>
  )
}

function CameraAssignForm({ stableId }: { stableId: string }) {
  const { sessionAssign } = useSetup()
  const [side, setSide] = useState<'left' | 'right' | ''>('')
  const [suffix, setSuffix] = useState('')

  const prefix = side ? `${side}_` : ''
  // Strip any leading left_/right_ the user re-typed so we never double-prefix.
  const cleanSuffix = suffix.trim().replace(/^(left_|right_)/, '')
  const finalAlias = side ? `${prefix}${cleanSuffix}` : ''
  const canSubmit = side !== '' && cleanSuffix.length > 0

  return (
    <div className={formBox}>
      <div className="flex gap-1.5">
        {(['left', 'right'] as const).map((s) => (
          <button key={s} onClick={() => setSide(s)}
            className={`px-2.5 py-1 text-2xs rounded-md border transition-colors ${
              side === s ? 'border-ac bg-ac/5 text-ac font-medium ring-1 ring-ac/20' : 'border-bd/50 text-tx2 hover:border-ac/50'
            }`}>
            {s === 'left' ? '左臂' : '右臂'}
          </button>
        ))}
      </div>
      <div className="flex items-center gap-1">
        <span className="text-xs text-tx2 select-none whitespace-nowrap">{prefix || 'left_/right_'}</span>
        <input value={suffix} onChange={(e) => setSuffix(e.target.value)}
          placeholder="wrist / front / ..." className={inputCls} disabled={!side} />
      </div>
      {finalAlias && (
        <p className="text-2xs text-tx2">预览: {finalAlias}</p>
      )}
      <button onClick={() => { if (side && cleanSuffix) sessionAssign(stableId, finalAlias, 'opencv', side) }}
        disabled={!canSubmit}
        className="px-3 py-1 text-xs bg-ac text-white rounded-lg font-medium disabled:opacity-40 hover:bg-ac2 transition-colors">
        确认分配
      </button>
    </div>
  )
}

function IdentifyStep() {
  const { t } = useI18n()
  const {
    catalog, selectedCategory, selectedModel,
    scannedPorts, scannedCameras, assignments,
    startMotion, stopMotion, goToStep, sessionUnassign,
  } = useSetup()
  const [activeId, setActiveId] = useState<string | null>(null)

  useEffect(() => {
    if (scannedPorts.length > 0) startMotion()
    return () => { stopMotion() }
  }, [])

  const roles = catalog?.models[selectedCategory]?.find((m) => m.name === selectedModel)?.roles ?? []
  const assignedIds = new Set(assignments.map((a) => a.interface_stable_id))
  const freePorts = scannedPorts.filter((p) => !assignedIds.has(p.stable_id))
  const freeCams = scannedCameras.filter((c) => !assignedIds.has(c.stable_id))

  return (
    <div className="space-y-4">
      <p className="text-sm text-tx2">{t('moveArmPrompt')}</p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Unassigned */}
        <div className="rounded-lg border border-bd/30 bg-white p-4 shadow-card">
          <h4 className="text-xs text-tx2 uppercase tracking-wider font-medium mb-2">未分配设备</h4>
          <div className="space-y-1.5">
            {freePorts.map((port) => (
              <div key={port.stable_id}>
                <div onClick={() => setActiveId(activeId === port.stable_id ? null : port.stable_id)}
                  className="cursor-pointer">
                  <DeviceNode id={`port:${port.stable_id}`} kind="port"
                    label={deviceLabel(port)} sublabel={`${port.motor_ids.length} motors`} moved={port.moved} />
                </div>
                {activeId === port.stable_id && port.moved && (
                  <PortAssignForm stableId={port.stable_id} roles={roles} model={selectedModel} />
                )}
              </div>
            ))}
            {freeCams.map((cam) => (
              <div key={cam.stable_id}>
                <div onClick={() => setActiveId(activeId === cam.stable_id ? null : cam.stable_id)}
                  className="cursor-pointer">
                  <DeviceNode id={`camera:${cam.stable_id}`} kind="camera"
                    label={deviceLabel(cam)} sublabel={`${cam.width}×${cam.height}`} previewUrl={cam.preview_url} />
                </div>
                {activeId === cam.stable_id && <CameraAssignForm stableId={cam.stable_id} />}
              </div>
            ))}
            {freePorts.length === 0 && freeCams.length === 0 && (
              <p className="text-sm text-tx2 py-2">所有设备已分配</p>
            )}
          </div>
        </div>
        {/* Assigned */}
        <div className="rounded-lg border border-gn/20 bg-gn/[0.02] p-4 shadow-card">
          <h4 className="text-xs text-tx2 uppercase tracking-wider font-medium mb-2">已分配设备</h4>
          <div className="space-y-1.5">
            {assignments.map((a) => (
              <div key={a.alias} className="group flex items-center gap-2 bg-white border border-gn/20 rounded-lg shadow-card px-3 py-2">
                <span className="w-2 h-2 rounded-full bg-gn shrink-0" />
                <span className="text-sm font-medium text-tx flex-1 truncate">{a.alias}</span>
                <span className="text-2xs text-tx2 shrink-0">{a.spec_name}</span>
                <button
                  onClick={() => sessionUnassign(a.alias)}
                  className="opacity-0 group-hover:opacity-100 transition-opacity text-sm text-tx3 hover:text-rd shrink-0"
                >
                  &times;
                </button>
              </div>
            ))}
            {assignments.length === 0 && <p className="text-sm text-tx2 py-2">暂无已分配设备</p>}
          </div>
        </div>
      </div>
      <div className="flex justify-between pt-2">
        <button onClick={async () => { await stopMotion(); goToStep('scan') }} className={btnBack}>返回</button>
        <button onClick={async () => { await stopMotion(); goToStep('review') }}
          disabled={assignments.length === 0} className={btnPrimary}>完成分配</button>
      </div>
    </div>
  )
}

// -- Step 4: Review ----------------------------------------------------------

function ReviewStep() {
  const { assignments, sessionUnassign, sessionCommit, goToStep } = useSetup()

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-bd/30 overflow-hidden shadow-card">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-tx2 text-xs bg-sf">
              <th className="px-3 py-2 font-medium">名称</th>
              <th className="px-3 py-2 font-medium">规格</th>
              <th className="px-3 py-2 font-medium">接口</th>
              <th className="px-3 py-2 font-medium w-12" />
            </tr>
          </thead>
          <tbody>
            {assignments.map((a) => (
              <tr key={a.alias} className="border-t border-bd/20">
                <td className="px-3 py-2 text-tx font-medium">{a.alias}</td>
                <td className="px-3 py-2 text-tx2">{a.spec_name}</td>
                <td className="px-3 py-2 text-tx2 text-2xs truncate max-w-[200px]">{a.interface_stable_id}</td>
                <td className="px-3 py-2">
                  <button onClick={() => sessionUnassign(a.alias)} className="text-2xs text-rd hover:text-rd/80">
                    移除
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {assignments.length === 0 && (
        <p className="text-sm text-tx2 text-center py-4">暂无分配，请返回添加设备</p>
      )}
      <div className="flex justify-between pt-2">
        <button onClick={() => goToStep('identify')} className={btnBack}>返回修改</button>
        <button onClick={sessionCommit} disabled={assignments.length === 0} className={btnPrimary}>
          确认提交
        </button>
      </div>
    </div>
  )
}

// -- Main Wizard -------------------------------------------------------------

export default function DiscoveryWizard() {
  const { wizardStep, error } = useSetup()

  return (
    <div className="rounded-xl border border-bd/30 bg-white p-8 space-y-4 shadow-elevated">
      {error && (
        <div className="rounded-lg border-l-4 border-l-rd bg-rd/5 border-y-0 border-r-0 p-3 text-sm text-rd">{error}</div>
      )}
      <StepIndicator current={wizardStep} />
      {wizardStep === 'select' && <ModelSelect />}
      {wizardStep === 'scan' && <ScanStep />}
      {wizardStep === 'identify' && <IdentifyStep />}
      {wizardStep === 'review' && <ReviewStep />}
    </div>
  )
}
