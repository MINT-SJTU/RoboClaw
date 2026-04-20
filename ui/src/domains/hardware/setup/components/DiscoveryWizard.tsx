import type { ReactNode } from 'react'
import { useEffect, useRef, useState } from 'react'
import { useSetup } from '@/domains/hardware/setup/store/useSetupStore'
import { useI18n } from '@/i18n'
import type {
  Assignment,
  CatalogModel,
  ScannedCamera,
  ScannedPort,
} from '@/domains/hardware/setup/store/useSetupStore'
import ScanArea from './ScanArea'
import PermissionPanel from './PermissionPanel'
import { cn } from '@/shared/lib/cn'
import {
  presentCameraLabel,
  presentCameraStream,
  presentPortLabel,
} from '@/domains/hardware/setup/lib/devicePresentation'

const STEPS = ['select', 'scan', 'identify', 'review'] as const
const btnBack = 'rounded-full px-4 py-2 text-sm font-medium text-tx2 transition-colors hover:text-tx'
const btnPrimary = 'rounded-full bg-ac px-4 py-2 text-sm font-semibold text-white transition-all hover:bg-ac2 disabled:cursor-not-allowed disabled:opacity-40'
const btnOutline = 'rounded-full border border-ac/40 px-4 py-2 text-sm font-medium text-ac transition-all hover:border-ac hover:bg-ac/5'
const inputCls = 'w-full rounded-xl border border-bd bg-white px-3 py-2.5 text-sm text-tx outline-none transition-all focus:border-ac focus:shadow-glow-ac'

function buildArmAlias(role: string, side: 'left' | 'right' | ''): string {
  if (!role) {
    return ''
  }
  return side ? `${side}_${role}` : role
}

function StepIndicator({ current }: { current: string }) {
  const idx = STEPS.indexOf(current as (typeof STEPS)[number])
  const labels = ['选择型号', '扫描', '分配', '确认']

  return (
    <div className="flex flex-wrap items-center justify-center gap-2">
      {STEPS.map((step, index) => (
        <div key={step} className="flex items-center gap-2">
          <div className={cn(
            'flex items-center gap-2 rounded-full px-3 py-1.5 text-2xs font-medium transition-colors',
            index === idx && 'bg-ac/10 text-ac',
            index < idx && 'text-gn',
            index > idx && 'text-tx2',
          )}>
            <span className={cn(
              'flex h-6 w-6 items-center justify-center rounded-full text-2xs',
              index < idx && 'bg-gn text-white',
              index === idx && 'bg-ac text-white',
              index > idx && 'bg-sf2 text-tx3',
            )}>
              {index < idx ? '✓' : index + 1}
            </span>
            <span>{labels[index]}</span>
          </div>
          {index < STEPS.length - 1 && (
            <div className={cn('h-px w-8', index < idx ? 'bg-gn/60' : 'bg-bd')} />
          )}
        </div>
      ))}
    </div>
  )
}

function WizardSection({
  title,
  description,
  children,
}: {
  title: string
  description: string
  children: ReactNode
}) {
  return (
    <section className="space-y-4">
      <div>
        <h3 className="text-sm font-bold uppercase tracking-[0.18em] text-tx">{title}</h3>
        <p className="mt-2 text-sm text-tx3">{description}</p>
      </div>
      {children}
    </section>
  )
}

function ChoiceButton({
  active,
  label,
  onClick,
}: {
  active: boolean
  label: string
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'rounded-xl border px-3 py-2 text-sm transition-all',
        active
          ? 'border-ac bg-ac/10 font-semibold text-ac'
          : 'border-bd/30 bg-white text-tx2 hover:border-ac/30 hover:text-ac',
      )}
    >
      {label}
    </button>
  )
}

function SetupFieldGroup({
  title,
  description,
  children,
}: {
  title: string
  description: string
  children: ReactNode
}) {
  return (
    <div className="rounded-2xl border border-bd/30 bg-white p-4">
      <div>
        <h4 className="text-sm font-semibold text-tx">{title}</h4>
        <p className="mt-1 text-sm text-tx3">{description}</p>
      </div>
      <div className="mt-4">{children}</div>
    </div>
  )
}

function SetupDeviceButton({
  label,
  sublabel,
  kind,
  moved,
  active,
  onClick,
  previewUrl,
}: {
  label: string
  sublabel: string
  kind: 'port' | 'camera'
  moved?: boolean
  active?: boolean
  onClick: () => void
  previewUrl?: string | null
}) {
  const { t } = useI18n()
  const icon = kind === 'port' ? '⊞' : '◎'

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'w-full rounded-2xl border px-3 py-3 text-left transition-all',
        active ? 'border-ac bg-ac/10 shadow-glow-ac' : 'border-bd/30 bg-white hover:border-ac/30',
      )}
    >
      <div className="flex items-center gap-3">
        {previewUrl ? (
          <img
            src={previewUrl}
            alt={label}
            className="h-11 w-14 shrink-0 rounded-xl border border-bd/30 object-cover"
            draggable={false}
          />
        ) : (
          <span className="flex h-11 w-14 shrink-0 items-center justify-center rounded-xl border border-bd/30 bg-sf2 text-sm text-tx2">
            {icon}
          </span>
        )}

        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-semibold text-tx">{label}</div>
          {sublabel && <div className="mt-1 truncate text-2xs text-tx2">{sublabel}</div>}
        </div>

        {kind === 'port' && (
          <span className={cn(
            'shrink-0 rounded-full px-2.5 py-1 text-2xs font-semibold',
            moved ? 'bg-gn/10 text-gn' : 'bg-yl/10 text-yl',
          )}>
            {moved ? t('setupMotionDetected') : t('setupMotionPending')}
          </span>
        )}
      </div>
    </button>
  )
}

function AssignmentCard({
  assignment,
  onRemove,
}: {
  assignment: Assignment
  onRemove: (alias: string) => void
}) {
  const { t } = useI18n()
  const specLabel = assignment.spec_name === 'opencv'
    ? t('camera')
    : assignment.spec_name
        .replace('_leader', ` · ${t('leader')}`)
        .replace('_follower', ` · ${t('follower')}`)

  return (
    <div className="rounded-2xl border border-gn/20 bg-white px-4 py-3 shadow-card">
      <div className="flex items-start gap-3">
        <span className="mt-1 h-2.5 w-2.5 shrink-0 rounded-full bg-gn" />
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-semibold text-tx">{assignment.alias}</div>
          <div className="mt-1 text-2xs text-tx2">{specLabel}</div>
        </div>
        <button
          type="button"
          onClick={() => onRemove(assignment.alias)}
          className="text-sm text-tx3 transition-colors hover:text-rd"
        >
          ×
        </button>
      </div>
    </div>
  )
}

function PortAssignForm({
  port,
  roles,
  model,
}: {
  port: ScannedPort
  roles: string[]
  model: string
}) {
  const { t } = useI18n()
  const { sessionAssign } = useSetup()
  const [role, setRole] = useState('')
  const [side, setSide] = useState<'left' | 'right' | ''>('')
  const alias = buildArmAlias(role, side)

  function submit() {
    if (!alias || !role) return
    sessionAssign(port.stable_id, alias, `${model}_${role}`, side)
  }

  const roleOptions = roles.map((value) => ({
    value,
    label: value === 'leader' ? t('leader') : value === 'follower' ? t('follower') : value,
  }))
  return (
    <div className="space-y-4">
      <SetupFieldGroup
        title={t('setupRoleTitle')}
        description={t('setupRoleDesc')}
      >
        <div className="flex flex-wrap gap-2">
          {roleOptions.map((option) => (
            <ChoiceButton
              key={option.value}
              active={role === option.value}
              label={option.label}
              onClick={() => setRole(option.value)}
            />
          ))}
        </div>
      </SetupFieldGroup>

      <SetupFieldGroup
        title={t('setupMountTitle')}
        description={t('setupMountDesc')}
      >
        <div className="flex flex-wrap gap-2">
          {[
            { value: '', label: t('setupSingleArm') },
            { value: 'left', label: t('setupLeftArm') },
            { value: 'right', label: t('setupRightArm') },
          ].map((option) => (
            <ChoiceButton
              key={option.label}
              active={side === option.value}
              label={option.label}
              onClick={() => setSide(option.value as 'left' | 'right' | '')}
            />
          ))}
        </div>
      </SetupFieldGroup>

      <SetupFieldGroup
        title={t('setupNameTitle')}
        description={t('setupNameDesc')}
      >
        <div className="rounded-xl border border-bd/30 bg-sf px-3 py-3 text-sm text-tx2">
          {t('setupAliasPreview')}: <span className="font-mono font-semibold text-tx">{alias || '—'}</span>
        </div>
      </SetupFieldGroup>

      <button
        type="button"
        onClick={submit}
        disabled={!alias || !role}
        className="w-full rounded-full bg-ac px-4 py-2.5 text-sm font-semibold text-white transition-all hover:bg-ac2 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {t('setupConfirmAssignment')}
      </button>
    </div>
  )
}

type CameraMount = 'left' | 'right' | 'single' | ''

function CameraAssignForm({ camera }: { camera: ScannedCamera }) {
  const { t } = useI18n()
  const { sessionAssign } = useSetup()
  const [mount, setMount] = useState<CameraMount>('')
  const [suffix, setSuffix] = useState('')

  const prefix = mount === 'left' || mount === 'right' ? `${mount}_` : ''
  const cleanSuffix = suffix.trim().replace(/^(left_|right_)/, '')
  const finalAlias = mount ? `${prefix}${cleanSuffix}` : ''
  const canSubmit = mount !== '' && cleanSuffix.length > 0

  function submit() {
    if (!canSubmit) return
    const side = mount === 'left' || mount === 'right' ? mount : ''
    sessionAssign(camera.stable_id, finalAlias, 'opencv', side)
  }

  return (
    <div className="space-y-4">
      <SetupFieldGroup
        title={t('setupMountTitle')}
        description={t('setupCameraMountDesc')}
      >
        <div className="flex flex-wrap gap-2">
          {[
            { value: 'left', label: t('setupLeftArm') },
            { value: 'right', label: t('setupRightArm') },
            { value: 'single', label: t('setupSingleArm') },
          ].map((option) => (
            <ChoiceButton
              key={option.value}
              active={mount === option.value}
              label={option.label}
              onClick={() => setMount(option.value as CameraMount)}
            />
          ))}
        </div>
      </SetupFieldGroup>

      <SetupFieldGroup
        title={t('setupNameTitle')}
        description={t('setupCameraNameDesc')}
      >
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            {prefix && (
              <span className="rounded-lg bg-sf px-2 py-2 text-sm font-mono text-tx2">
                {prefix}
              </span>
            )}
            <input
              value={suffix}
              onChange={(event) => setSuffix(event.target.value)}
              placeholder={mount === 'single' ? t('camera') : 'wrist / front / ...'}
              className={inputCls}
              disabled={!mount}
            />
          </div>
          <div className="text-sm text-tx3">
            {t('setupAliasPreview')}: <span className="font-mono text-tx">{finalAlias || '—'}</span>
          </div>
        </div>
      </SetupFieldGroup>

      <button
        type="button"
        onClick={submit}
        disabled={!canSubmit}
        className="w-full rounded-full bg-ac px-4 py-2.5 text-sm font-semibold text-white transition-all hover:bg-ac2 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {t('setupConfirmAssignment')}
      </button>
    </div>
  )
}

function ModelSelect() {
  const { t } = useI18n()
  const {
    catalog,
    selectedCategory,
    selectedModel,
    setCategory,
    setModel,
    goToStep,
    cancelWizard,
  } = useSetup()
  const categories = catalog?.categories ?? []
  const models: CatalogModel[] = selectedCategory && catalog?.models
    ? catalog.models[selectedCategory] ?? []
    : []

  const categoryIcons: Record<string, string> = { arm: '🦾', hand: '🤚', humanoid: '🤖', mobile: '🚗' }
  const categoryLabels: Record<string, string> = {
    arm: t('arm'),
    hand: t('hand'),
    humanoid: t('humanoid'),
    mobile: t('mobile'),
  }

  return (
    <div className="space-y-6">
      <WizardSection
        title={t('selectEmbodimentType')}
        description={t('setupChooseTypeHint')}
      >
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {categories.map((category) => (
            <button
              key={category.id}
              type="button"
              disabled={!category.supported}
              onClick={() => setCategory(category.id)}
              className={cn(
                'rounded-2xl border px-4 py-5 text-left transition-all',
                !category.supported && 'cursor-not-allowed border-bd/20 bg-sf text-tx3 opacity-60',
                category.supported && category.id === selectedCategory && 'border-ac bg-ac/5 text-ac shadow-glow-ac',
                category.supported && category.id !== selectedCategory && 'border-bd/30 bg-white text-tx hover:border-ac/30',
              )}
            >
              <div className="text-2xl">{categoryIcons[category.id] || '📦'}</div>
              <div className="mt-4 text-base font-semibold">{categoryLabels[category.id] || category.id}</div>
              {!category.supported && (
                <div className="mt-2 text-sm text-tx3">{t('notSupportedYet')}</div>
              )}
            </button>
          ))}
        </div>
      </WizardSection>

      {selectedCategory && models.length > 0 && (
        <WizardSection
          title={t('selectModel', { n: categoryLabels[selectedCategory] || selectedCategory })}
          description={t('setupChooseModelHint')}
        >
          <div className="grid gap-3 md:grid-cols-2">
            {models.map((model) => (
              <button
                key={model.name}
                type="button"
                onClick={() => {
                  setModel(model.name)
                  goToStep('scan')
                }}
                className={cn(
                  'rounded-2xl border px-4 py-4 text-left transition-all',
                  model.name === selectedModel
                    ? 'border-ac bg-ac/10 text-ac shadow-glow-ac'
                    : 'border-bd/30 bg-white text-tx hover:border-ac/30',
                )}
              >
                <div className="text-lg font-semibold">{model.name}</div>
              </button>
            ))}
          </div>
        </WizardSection>
      )}

      <div className="flex justify-end">
        <button type="button" onClick={() => { void cancelWizard() }} className={btnBack}>
          {t('cancel')}
        </button>
      </div>
    </div>
  )
}

function ScanSummarySection({
  title,
  items,
  empty,
}: {
  title: string
  items: string[]
  empty: string
}) {
  return (
    <div className="rounded-2xl border border-bd/30 bg-white p-4">
      <h4 className="text-sm font-semibold text-tx">{title}</h4>
      <div className="mt-3 space-y-2">
        {items.length > 0 && items.map((item) => (
          <div key={item} className="rounded-xl bg-sf px-3 py-2 text-sm text-tx2">
            {item}
          </div>
        ))}
        {items.length === 0 && (
          <div className="rounded-xl border border-dashed border-bd/40 px-3 py-4 text-sm text-tx3">
            {empty}
          </div>
        )}
      </div>
    </div>
  )
}

function ScanStep() {
  const { t } = useI18n()
  const {
    scannedPorts,
    scannedCameras,
    scanning,
    doScan,
    goToStep,
    checkPermissions,
  } = useSetup()
  const [permChecked, setPermChecked] = useState(false)

  useEffect(() => {
    checkPermissions().then((permissions) => {
      setPermChecked(true)
      if (!permissions || (permissions.serial.ok && permissions.camera.ok)) {
        doScan()
      }
    })
  }, [checkPermissions, doScan])

  const hasDevices = scannedPorts.length > 0 || scannedCameras.length > 0

  return (
    <div className="space-y-5">
      {permChecked && <PermissionPanel bare onFixed={() => { void doScan() }} />}

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1.15fr)_minmax(280px,0.85fr)]">
        <div className="rounded-2xl border border-bd/30 bg-white p-5 shadow-card">
          <WizardSection
            title={t('scanResultsTitle')}
            description={t('settingsSetupDesc')}
          >
            <ScanArea ports={scannedPorts} cameras={scannedCameras} scanning={scanning} />
          </WizardSection>
        </div>

        <div className="space-y-4">
          <ScanSummarySection
            title={t('scanArmsTitle')}
            items={scannedPorts.map((port) => presentPortLabel(port))}
            empty={t('setupNoPortsDetected')}
          />
          <ScanSummarySection
            title={t('scanCamerasTitle')}
            items={scannedCameras.map((camera) => presentCameraLabel(camera, scannedCameras, t))}
            empty={t('setupNoCamerasDetected')}
          />
        </div>
      </div>

      <div className="flex justify-between pt-2">
        <button type="button" onClick={() => goToStep('select')} className={btnBack}>
          {t('cancel')}
        </button>
        <div className="flex gap-2">
          {!scanning && (
            <button type="button" onClick={() => { void doScan() }} className={btnOutline}>
              {t('refresh')}
            </button>
          )}
          {!scanning && hasDevices && (
            <button type="button" onClick={() => goToStep('identify')} className={btnPrimary}>
              {t('setupNextStep')}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function IdentifyStep() {
  const { t } = useI18n()
  const {
    catalog,
    selectedCategory,
    selectedModel,
    scannedPorts,
    scannedCameras,
    assignments,
    startMotion,
    stopMotion,
    goToStep,
    sessionDismiss,
    sessionUnassign,
  } = useSetup()
  const [activeId, setActiveId] = useState<string | null>(null)

  const selectedCatalogModel = catalog?.models[selectedCategory]?.find((model) => model.name === selectedModel) ?? null
  const roles = selectedCatalogModel?.roles ?? []
  const assignedIds = new Set(assignments.map((assignment) => assignment.interface_stable_id))
  const freePorts = scannedPorts.filter((port) => !assignedIds.has(port.stable_id))
  const freeCameras = scannedCameras.filter((camera) => !assignedIds.has(camera.stable_id))
  const pendingIds = [...freePorts.map((port) => port.stable_id), ...freeCameras.map((camera) => camera.stable_id)]
  const detectedPortId = freePorts.find((port) => port.moved)?.stable_id ?? null
  const lastDetectedPortIdRef = useRef<string | null>(null)

  useEffect(() => {
    if (freePorts.length > 0) {
      void startMotion()
      return () => { void stopMotion() }
    }
    void stopMotion()
    return undefined
  }, [freePorts.length, startMotion, stopMotion])

  useEffect(() => {
    if (pendingIds.length === 0) {
      if (activeId) setActiveId(null)
      return
    }
    if (!activeId || !pendingIds.includes(activeId)) {
      setActiveId(pendingIds[0])
    }
  }, [activeId, pendingIds])

  const activePort = freePorts.find((port) => port.stable_id === activeId) || null
  const activeCamera = freeCameras.find((camera) => camera.stable_id === activeId) || null

  useEffect(() => {
    if (!detectedPortId) {
      lastDetectedPortIdRef.current = null
      return
    }
    const detectionChanged = lastDetectedPortIdRef.current !== detectedPortId
    const selectedNonDetectedPort = Boolean(
      activePort && activePort.stable_id !== detectedPortId && !activePort.moved,
    )
    if (activeCamera && !detectionChanged) {
      return
    }
    if (!detectionChanged && !selectedNonDetectedPort) {
      return
    }
    lastDetectedPortIdRef.current = detectedPortId
    setActiveId(detectedPortId)
  }, [activeCamera, activePort, detectedPortId])

  const selectedLabel = activePort
    ? presentPortLabel(activePort)
    : activeCamera
      ? presentCameraLabel(activeCamera, freeCameras, t)
      : ''
  const activeStableId = activePort?.stable_id || activeCamera?.stable_id || ''

  return (
    <div className="space-y-5">
      <div className="rounded-2xl border border-ac/20 bg-ac/5 px-4 py-3 text-sm text-tx2">
        {t('setupAssignmentDesc')}
      </div>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
        <div className="space-y-4">
          <section className="rounded-2xl border border-bd/30 bg-white p-4 shadow-card">
            <h4 className="text-sm font-semibold text-tx">{t('setupPendingDevices')}</h4>
            <div className="mt-4 space-y-3">
              {freePorts.map((port) => (
                <SetupDeviceButton
                  key={port.stable_id}
                  label={presentPortLabel(port)}
                  sublabel={`${port.motor_ids.length} ${t('motorsFound')}`}
                  kind="port"
                  moved={port.moved}
                  active={activeId === port.stable_id}
                  onClick={() => setActiveId(port.stable_id)}
                />
              ))}
              {freeCameras.map((camera) => (
                <SetupDeviceButton
                  key={camera.stable_id}
                  label={presentCameraLabel(camera, freeCameras, t)}
                  sublabel={presentCameraStream(camera, t)}
                  kind="camera"
                  active={activeId === camera.stable_id}
                  onClick={() => setActiveId(camera.stable_id)}
                  previewUrl={camera.preview_url}
                />
              ))}
              {freePorts.length === 0 && freeCameras.length === 0 && (
                <div className="rounded-xl border border-dashed border-bd/40 px-3 py-4 text-sm text-tx3">
                  {t('setupNoPendingDevices')}
                </div>
              )}
            </div>
          </section>

          <section className="rounded-2xl border border-gn/20 bg-gn/[0.04] p-4 shadow-card">
            <h4 className="text-sm font-semibold text-tx">{t('setupAssignedDevices')}</h4>
            <div className="mt-4 space-y-3">
              {assignments.map((assignment) => (
                <AssignmentCard
                  key={assignment.alias}
                  assignment={assignment}
                  onRemove={(alias) => { void sessionUnassign(alias) }}
                />
              ))}
              {assignments.length === 0 && (
                <div className="rounded-xl border border-dashed border-bd/40 px-3 py-4 text-sm text-tx3">
                  {t('setupNoAssignedDevices')}
                </div>
              )}
            </div>
          </section>
        </div>

        <section className="rounded-2xl border border-bd/30 bg-sf p-5 shadow-card">
          <div className="border-b border-bd/30 pb-4">
            <div className="text-2xs font-semibold uppercase tracking-[0.18em] text-tx3">
              {t('setupSelectedDevice')}
            </div>
            <h3 className="mt-2 text-xl font-semibold text-tx">
              {selectedLabel || t('setupSelectDevicePrompt')}
            </h3>
            {activePort && (
              <p className="mt-2 text-sm text-tx3">
                {activePort.moved ? t('setupPortDetectedHint') : t('moveArmPrompt')}
              </p>
            )}
            {activeCamera && (
              <p className="mt-2 text-sm text-tx3">{t('setupCameraDetectedHint')}</p>
            )}
          </div>

          <div className="mt-5">
            {activePort && activePort.moved && (
              <PortAssignForm port={activePort} roles={roles} model={selectedModel} />
            )}
            {activePort && !activePort.moved && (
              <div className="rounded-2xl border border-dashed border-yl/30 bg-white px-4 py-6 text-sm text-tx3">
                {t('moveArmPrompt')}
              </div>
            )}
            {activeCamera && <CameraAssignForm camera={activeCamera} />}
            {!activePort && !activeCamera && (
              <div className="rounded-2xl border border-dashed border-bd/40 bg-white px-4 py-6 text-sm text-tx3">
                {assignments.length > 0 ? t('setupReadyToCommit') : t('setupSelectDevicePrompt')}
              </div>
            )}
            {(activePort || activeCamera) && (
              <div className="mt-4 flex justify-end">
                <button
                  type="button"
                  onClick={() => { void sessionDismiss(activeStableId) }}
                  className={btnOutline}
                >
                  {t('setupSkipDevice')}
                </button>
              </div>
            )}
          </div>
        </section>
      </div>

      <div className="flex justify-between pt-2">
        <button
          type="button"
          onClick={() => {
            void stopMotion()
            goToStep('scan')
          }}
          className={btnBack}
        >
          {t('cancel')}
        </button>
        <button
          type="button"
          onClick={() => {
            void stopMotion()
            goToStep('review')
          }}
          disabled={assignments.length === 0}
          className={btnPrimary}
        >
          {t('setupReviewStep')}
        </button>
      </div>
    </div>
  )
}

function ReviewStep() {
  const { t } = useI18n()
  const { assignments, sessionUnassign, sessionCommit, goToStep } = useSetup()

  return (
    <div className="space-y-5">
      <div className="rounded-2xl border border-bd/30 bg-white p-5 shadow-card">
        <WizardSection
          title={t('setupReviewTitle')}
          description={t('setupReviewDesc')}
        >
          <div className="grid gap-3 md:grid-cols-2">
            {assignments.map((assignment) => (
              <div key={assignment.alias} className="rounded-2xl border border-bd/30 bg-sf p-4">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="truncate text-base font-semibold text-tx">{assignment.alias}</div>
                    <div className="mt-2 text-sm text-tx2">{assignment.spec_name}</div>
                    <div className="mt-1 text-2xs text-tx3">
                      {t('setupInterface')}: {assignment.interface_stable_id}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => { void sessionUnassign(assignment.alias) }}
                    className="text-sm text-tx3 transition-colors hover:text-rd"
                  >
                    ×
                  </button>
                </div>
              </div>
            ))}
          </div>
          {assignments.length === 0 && (
            <div className="rounded-xl border border-dashed border-bd/40 px-3 py-4 text-sm text-tx3">
              {t('setupNoAssignedDevices')}
            </div>
          )}
        </WizardSection>
      </div>

      <div className="flex justify-between pt-2">
        <button type="button" onClick={() => goToStep('identify')} className={btnBack}>
          {t('setupBackToAssignment')}
        </button>
        <button
          type="button"
          onClick={() => { void sessionCommit() }}
          disabled={assignments.length === 0}
          className={btnPrimary}
        >
          {t('setupCommit')}
        </button>
      </div>
    </div>
  )
}

export default function DiscoveryWizard() {
  const { wizardStep, error } = useSetup()

  return (
    <div className="space-y-5">
      {error && (
        <div className="rounded-2xl border border-rd/30 border-l-4 border-l-rd bg-rd/5 p-3 text-sm text-rd">
          {error}
        </div>
      )}
      <StepIndicator current={wizardStep} />
      {wizardStep === 'select' && <ModelSelect />}
      {wizardStep === 'scan' && <ScanStep />}
      {wizardStep === 'identify' && <IdentifyStep />}
      {wizardStep === 'review' && <ReviewStep />}
    </div>
  )
}
