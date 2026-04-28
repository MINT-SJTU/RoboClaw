import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import AppShell from '@/app/shell/AppShell'
import ControlPage from '@/domains/control/pages/ControlPage'
import RecoveryCenterPage from '@/domains/recovery/pages/RecoveryCenterPage'
import DatasetExplorerPage from '@/domains/datasets/explorer/pages/DatasetExplorerPage'
import TrainingCenterPage from '@/domains/training/pages/TrainingCenterPage'
import QualityValidationPage from '@/domains/curation/quality/pages/QualityValidationPage'
import TextAlignmentPage from '@/domains/curation/text-alignment/pages/TextAlignmentPage'
import SettingsOverviewPage from '@/domains/settings/pages/SettingsOverviewPage'
import HardwareSettingsPage from '@/domains/settings/pages/HardwareSettingsPage'
import ProviderSettingsPage from '@/domains/settings/pages/ProviderSettingsPage'
import HubSettingsPage from '@/domains/settings/pages/HubSettingsPage'
import LogsPage from '@/domains/logs/pages/LogsPage'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<AppShell />}>
          <Route index element={<Navigate to="/control" replace />} />
          <Route path="control" element={<ControlPage />} />
          <Route path="recovery" element={<RecoveryCenterPage />} />
          <Route path="datasets" element={<Navigate to="/curation/datasets" replace />} />
          <Route path="datasets/explorer" element={<Navigate to="/curation/datasets" replace />} />
          <Route path="training" element={<TrainingCenterPage />} />
          <Route path="curation" element={<Navigate to="/curation/datasets" replace />} />
          <Route path="curation/datasets" element={<DatasetExplorerPage />} />
          <Route path="curation/datasets/explorer" element={<Navigate to="/curation/datasets" replace />} />
          <Route path="curation/quality" element={<QualityValidationPage />} />
          <Route path="curation/text-alignment" element={<TextAlignmentPage />} />
          <Route path="settings" element={<SettingsOverviewPage />} />
          <Route path="settings/hardware" element={<HardwareSettingsPage />} />
          <Route path="settings/provider" element={<ProviderSettingsPage />} />
          <Route path="settings/hub" element={<HubSettingsPage />} />
          <Route path="logs" element={<LogsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
