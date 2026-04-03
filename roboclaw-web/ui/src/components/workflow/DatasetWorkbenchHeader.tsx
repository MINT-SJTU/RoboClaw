import { useEffect, useState } from 'react'
import { useI18n } from '../../controllers/i18n'
import { useWorkflow } from '../../controllers/workflow'
import { ActionButton } from '../ux'

export default function DatasetWorkbenchHeader() {
  const { t } = useI18n()
  const {
    datasets,
    datasetsLoading,
    selectedDataset,
    selectDataset,
    importDatasetFromHf,
    datasetImportJob,
  } = useWorkflow()
  const [datasetIdInput, setDatasetIdInput] = useState('')
  const [importError, setImportError] = useState('')

  useEffect(() => {
    if (datasetsLoading || selectedDataset || datasets.length !== 1) {
      return
    }
    void selectDataset(datasets[0].name)
  }, [datasets, datasetsLoading, selectedDataset, selectDataset])

  async function handleImport(): Promise<void> {
    const datasetId = datasetIdInput.trim()
    if (!datasetId) {
      return
    }
    setImportError('')
    try {
      await importDatasetFromHf(datasetId, true)
      setDatasetIdInput('')
    } catch (error) {
      setImportError(error instanceof Error ? error.message : 'Dataset import failed')
    }
  }

  return (
    <div className="dataset-workbench">
      <div className="dataset-workbench__controls">
        <label className="dataset-workbench__control">
          <span>{t('selectDataset')}</span>
          <select
            className="dataset-workbench__select"
            value={selectedDataset ?? ''}
            onChange={(event) => {
              if (event.target.value) {
                void selectDataset(event.target.value)
              }
            }}
            disabled={datasetsLoading}
          >
            <option value="">
              {datasetsLoading ? t('running') : t('selectDataset')}
            </option>
            {datasets.map((dataset) => (
              <option key={dataset.name} value={dataset.name}>
                {dataset.name}
              </option>
            ))}
          </select>
        </label>

        <label className="dataset-workbench__control dataset-workbench__control--wide">
          <span>{t('hfDatasetId')}</span>
          <input
            className="dataset-workbench__input"
            type="text"
            value={datasetIdInput}
            onChange={(event) => setDatasetIdInput(event.target.value)}
            placeholder={t('hfDatasetPlaceholder')}
          />
        </label>

        <ActionButton
          type="button"
          variant="secondary"
          onClick={() => void handleImport()}
          disabled={datasetImportJob?.status === 'queued' || datasetImportJob?.status === 'running'}
          className="dataset-workbench__import-btn"
        >
          {datasetImportJob?.status === 'queued' || datasetImportJob?.status === 'running'
            ? t('importingDataset')
            : t('importDataset')}
        </ActionButton>
      </div>

      {(datasetImportJob || importError) && (
        <div className={`dataset-workbench__status ${datasetImportJob?.status === 'error' || importError ? 'is-error' : ''}`}>
          {importError
            || datasetImportJob?.message
            || t('importQueued')}
        </div>
      )}
    </div>
  )
}
