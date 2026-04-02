import { useState } from 'react'
import { DndContext, DragEndEvent, DragOverlay, DragStartEvent } from '@dnd-kit/core'
import { useSetup } from '../controllers/setup'
import { useI18n } from '../controllers/i18n'
import ScanArea from '../components/setup/ScanArea'
import SetupZone from '../components/setup/SetupZone'
import DeviceNode from '../components/setup/DeviceNode'

export default function SetupWizardModal() {
  const store = useSetup()
  const { t } = useI18n()
  const [activeDrag, setActiveDrag] = useState<string | null>(null)
  const [showAddDialog, setShowAddDialog] = useState<{ kind: 'port' | 'camera'; id: string } | null>(null)
  const [alias, setAlias] = useState('')
  const [armType, setArmType] = useState<'so101_leader' | 'so101_follower'>('so101_follower')

  if (!store.open) return null

  const handleDragStart = (event: DragStartEvent) => {
    setActiveDrag(String(event.active.id))
  }

  const handleDragEnd = (event: DragEndEvent) => {
    setActiveDrag(null)
    const { active, over } = event
    if (!over || over.id !== 'setup-zone') return
    const id = String(active.id)
    if (id.startsWith('port:')) {
      setShowAddDialog({ kind: 'port', id: id.replace('port:', '') })
      setAlias('')
    } else if (id.startsWith('camera:')) {
      setShowAddDialog({ kind: 'camera', id: id.replace('camera:', '') })
      setAlias('')
    }
  }

  const handleAddConfirm = async () => {
    if (!showAddDialog || !alias.trim()) return
    if (showAddDialog.kind === 'port') {
      await store.addArm(alias.trim(), armType, showAddDialog.id)
    } else {
      const idx = parseInt(showAddDialog.id, 10)
      await store.addCamera(alias.trim(), idx)
    }
    setShowAddDialog(null)
  }

  // Find the dragged node data for overlay
  const draggedPort = activeDrag?.startsWith('port:')
    ? store.scannedPorts.find((p) => `port:${p.port_id}` === activeDrag)
    : null
  const draggedCam = activeDrag?.startsWith('camera:')
    ? store.scannedCameras.find((c) => `camera:${c.index}` === activeDrag)
    : null

  return (
    <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-bg border border-bd rounded-2xl shadow-2xl w-full max-w-5xl max-h-[90vh] flex flex-col animate-node-appear">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-bd">
          <h2 className="text-lg font-semibold text-tx">{t('setupWizard')}</h2>
          <button
            onClick={() => store.setOpen(false)}
            className="w-8 h-8 flex items-center justify-center rounded-full hover:bg-bd/30 text-tx2 text-lg"
          >
            ✕
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-6">
          <DndContext onDragStart={handleDragStart} onDragEnd={handleDragEnd}>
            <div className="flex gap-6 min-h-[400px]">
              {/* Left: Scan area */}
              <ScanArea
                ports={store.scannedPorts}
                cameras={store.scannedCameras}
                scanning={store.scanning}
              />

              {/* Right: Setup zone */}
              <SetupZone
                arms={store.configuredArms}
                cameras={store.configuredCameras}
                onRemoveArm={store.removeArm}
                onRemoveCamera={store.removeCamera}
              />
            </div>

            <DragOverlay>
              {draggedPort && (
                <DeviceNode
                  id={`port:${draggedPort.port_id}`}
                  kind="port"
                  label={draggedPort.by_id?.split('/').pop() || draggedPort.dev}
                  sublabel={`${draggedPort.motor_ids.length} ${t('motorsFound')}`}
                  moved={draggedPort.moved}
                />
              )}
              {draggedCam && (
                <DeviceNode
                  id={`camera:${draggedCam.index}`}
                  kind="camera"
                  label={draggedCam.by_id?.split('/').pop() || draggedCam.dev}
                  sublabel={`${draggedCam.width}×${draggedCam.height}`}
                />
              )}
            </DragOverlay>
          </DndContext>

          {store.error && (
            <div className="mt-4 px-4 py-2 bg-rd/10 border border-rd/30 rounded-lg text-rd text-sm">
              {store.error}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center gap-3 px-6 py-4 border-t border-bd">
          <button
            onClick={store.doScan}
            disabled={store.scanning}
            className="px-4 py-2 bg-ac text-white rounded-lg text-sm font-medium hover:bg-ac/90 disabled:opacity-50 transition-colors"
          >
            {store.scanning ? t('scanning') : t('scanDevices')}
          </button>
          <button
            onClick={store.motionActive ? store.stopMotion : store.startMotion}
            disabled={store.scannedPorts.length === 0}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              store.motionActive
                ? 'bg-yl/10 text-yl border border-yl hover:bg-yl/20'
                : 'bg-gn/10 text-gn border border-gn hover:bg-gn/20 disabled:opacity-50'
            }`}
          >
            {store.motionActive ? t('stopDetection') : t('detectMotion')}
          </button>
          <div className="flex-1" />
          <button
            onClick={() => store.setOpen(false)}
            className="px-4 py-2 bg-gn text-white rounded-lg text-sm font-medium hover:bg-gn/90 transition-colors"
          >
            {t('saveAndClose')}
          </button>
        </div>
      </div>

      {/* Add device dialog */}
      {showAddDialog && (
        <div className="fixed inset-0 z-60 flex items-center justify-center bg-black/40">
          <div className="bg-bg border border-bd rounded-xl shadow-xl p-6 w-[360px] animate-node-appear">
            <h3 className="text-sm font-semibold text-tx mb-4">
              {showAddDialog.kind === 'port' ? t('assignType') : t('assignAlias')}
            </h3>
            <label className="block text-xs text-tx2 mb-1">{t('assignAlias')}</label>
            <input
              value={alias}
              onChange={(e) => setAlias(e.target.value)}
              placeholder="e.g. left_arm"
              autoFocus
              className="w-full bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm mb-3 focus:outline-none focus:border-ac"
              onKeyDown={(e) => e.key === 'Enter' && handleAddConfirm()}
            />
            {showAddDialog.kind === 'port' && (
              <>
                <label className="block text-xs text-tx2 mb-1">{t('assignType')}</label>
                <div className="flex gap-2 mb-4">
                  <button
                    onClick={() => setArmType('so101_leader')}
                    className={`flex-1 py-2 rounded-lg text-sm border transition-colors ${
                      armType === 'so101_leader' ? 'border-ac bg-ac/10 text-ac' : 'border-bd text-tx2'
                    }`}
                  >
                    {t('leader')}
                  </button>
                  <button
                    onClick={() => setArmType('so101_follower')}
                    className={`flex-1 py-2 rounded-lg text-sm border transition-colors ${
                      armType === 'so101_follower' ? 'border-gn bg-gn/10 text-gn' : 'border-bd text-tx2'
                    }`}
                  >
                    {t('follower')}
                  </button>
                </div>
              </>
            )}
            <div className="flex gap-2">
              <button
                onClick={() => setShowAddDialog(null)}
                className="flex-1 py-2 border border-bd rounded-lg text-sm text-tx2 hover:bg-bd/30"
              >
                Cancel
              </button>
              <button
                onClick={handleAddConfirm}
                disabled={!alias.trim()}
                className="flex-1 py-2 bg-ac text-white rounded-lg text-sm font-medium hover:bg-ac/90 disabled:opacity-50"
              >
                OK
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
