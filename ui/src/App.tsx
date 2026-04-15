import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import ControlView from './views/ControlView'
import DataView from './views/DataView'
import DatasetExplorerView from './views/DatasetExplorerView'
import LogView from './views/LogView'
import QualityValidationView from './views/QualityValidationView'
import ChatView from './views/ChatView'
import SettingsView from './views/SettingsView'
import TextAlignmentView from './views/TextAlignmentView'
import WorkflowView from './views/WorkflowView'
import Layout from './components/Layout'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/control" replace />} />
          <Route path="control" element={<ControlView />} />
          <Route path="data" element={<DataView />} />
          <Route path="explorer" element={<DatasetExplorerView />} />
          <Route path="quality" element={<QualityValidationView />} />
          <Route path="text-alignment" element={<TextAlignmentView />} />
          <Route path="workflow" element={<WorkflowView />} />
          <Route path="settings" element={<SettingsView />} />
          <Route path="logs" element={<LogView />} />
          <Route path="chat" element={<ChatView />} />
          <Route path="dashboard" element={<Navigate to="/control" replace />} />
          <Route path="setup" element={<Navigate to="/settings" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
