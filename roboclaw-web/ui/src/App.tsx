import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import ChatView from './views/ChatView'
import SettingsView from './views/SettingsView'
import DashboardView from './views/DashboardView'
import QualityValidationView from './views/QualityValidationView'
import TextAlignmentView from './views/TextAlignmentView'
import WorkflowView from './views/WorkflowView'
import DatasetExplorerView from './views/DatasetExplorerView'
import Layout from './components/Layout'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<DashboardView />} />
          <Route path="quality" element={<QualityValidationView />} />
          <Route path="text-alignment" element={<TextAlignmentView />} />
          <Route path="explorer" element={<DatasetExplorerView />} />
          <Route path="workflow" element={<WorkflowView />} />
          <Route path="chat" element={<ChatView />} />
          <Route path="settings" element={<SettingsView />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
