import { create } from 'zustand'

export type Locale = 'zh' | 'en'

const translations = {
  zh: {
    // Header
    chat: '对话',
    dataCollection: '数据采集',
    settings: '设置',
    connected: '已连接',
    disconnected: '未连接',

    // Chat
    startChat: '开始与 RoboClaw 对话',
    inputPlaceholder: '输入消息...',
    waitingConnection: '等待连接...',
    send: '发送',
    providerWarning: '当前还没有配置可用的 provider。请先到',
    settingsPage: '设置',
    providerWarningEnd: '页面填写 API key 或 API base，保存后新的聊天请求会立即使用更新后的配置。',

    // Settings
    settingsTitle: '设置',
    settingsDesc: '这里只保留整个 RoboClaw 实例级别的全局 provider 配置。填写 base URL 和 API key 后，新的对话请求会直接使用这份配置。',
    globalProvider: 'Global Provider',
    baseUrl: 'Base URL',
    apiKey: 'API Key',
    apiKeyPlaceholder: '留空表示保持当前 key 不变',
    savedStatus: '当前保存状态',
    saved: '已保存',
    notSaved: '未保存',
    savedKey: '已保存的 key',
    saveSettings: '保存设置',
    saving: '保存中...',
    saveRedirectHint: '保存后会自动跳回聊天页。',
    saveSuccess: 'Provider settings saved. New chat requests will use this global RoboClaw provider.',
    loading: 'Loading provider settings...',

    // Data Collection
    connect: '连接',
    disconnect: '断开',
    connection: '连接',
    teleoperation: '遥操作',
    startTeleop: '开始遥操作',
    stopTeleop: '停止遥操作',
    recording: '录制',
    datasetName: '数据集名称',
    taskDesc: '任务描述',
    numEpisodes: '回合数',
    startRecording: '开始录制',
    stopRecording: '停止录制',
    saveEpisode: '保存回合',
    discardEpisode: '重置回合',
    episodesRecorded: '已录制回合',
    datasets: '数据集',
    refresh: '刷新',
    noDatasets: '暂无数据集',
    log: '日志',
    clear: '清除',
    noCameraFeed: '无相机画面',
    del: '删除',
    deleteConfirm: '确定删除数据集',

    // States
    stateDisconnected: '未连接',
    stateConnected: '已连接',
    stateTeleoperating: '遥操作中',
    stateRecording: '录制中',

    // Hardware
    hwReady: '硬件就绪',
    hwNotReady: '硬件未就绪',
    hwUncalibrated: '未校准',
    arms: '机械臂',
    cameras: '摄像头',
    hwConnected: '已连接',
    hwDisconnected: '未连接',
    hwCalibrated: '已校准',
    noArms: '未检测到机械臂',
    noCameras: '未检测到摄像头',
    leader: '主动臂',
    follower: '从动臂',
    enablePreview: '开启预览',
    disablePreview: '关闭预览',
    camerasDisabled: '摄像头预览已关闭',
    connecting: '连接中...',
    startingTeleop: '启动遥操作中...',
    startingRecord: '启动录制中...',
    hwInitializing: '硬件初始化中，请稍候',
    fillDatasetName: '请填写数据集名称',
    fillTaskDesc: '请填写任务描述',
    servoPositions: '舵机位置',
    servoLoading: '正在读取舵机数据...',
    servoBusy: '串口占用中，遥操作/录制结束后恢复',
  },
  en: {
    // Header
    chat: 'Chat',
    dataCollection: 'Data Collection',
    settings: 'Settings',
    connected: 'Connected',
    disconnected: 'Disconnected',

    // Chat
    startChat: 'Start chatting with RoboClaw',
    inputPlaceholder: 'Type a message...',
    waitingConnection: 'Waiting for connection...',
    send: 'Send',
    providerWarning: 'No provider configured. Go to',
    settingsPage: 'Settings',
    providerWarningEnd: 'to fill in API key or API base. New chat requests will use the updated configuration immediately.',

    // Settings
    settingsTitle: 'Settings',
    settingsDesc: 'Global provider configuration for this RoboClaw instance. Fill in the base URL and API key, and new chat requests will use this configuration.',
    globalProvider: 'Global Provider',
    baseUrl: 'Base URL',
    apiKey: 'API Key',
    apiKeyPlaceholder: 'Leave empty to keep current key',
    savedStatus: 'Current status',
    saved: 'Saved',
    notSaved: 'Not saved',
    savedKey: 'Saved key',
    saveSettings: 'Save Settings',
    saving: 'Saving...',
    saveRedirectHint: 'Will redirect to chat after saving.',
    saveSuccess: 'Provider settings saved. New chat requests will use this global RoboClaw provider.',
    loading: 'Loading provider settings...',

    // Data Collection
    connect: 'Connect',
    disconnect: 'Disconnect',
    connection: 'Connection',
    teleoperation: 'Teleoperation',
    startTeleop: 'Start Teleop',
    stopTeleop: 'Stop Teleop',
    recording: 'Recording',
    datasetName: 'Dataset Name',
    taskDesc: 'Task Description',
    numEpisodes: 'Num Episodes',
    startRecording: 'Start Recording',
    stopRecording: 'Stop Recording',
    saveEpisode: 'Save Episode',
    discardEpisode: 'Reset Episode',
    episodesRecorded: 'Episodes Recorded',
    datasets: 'Datasets',
    refresh: 'Refresh',
    noDatasets: 'No datasets',
    log: 'Log',
    clear: 'Clear',
    noCameraFeed: 'No camera feed',
    del: 'Del',
    deleteConfirm: 'Delete dataset',

    // States
    stateDisconnected: 'Disconnected',
    stateConnected: 'Connected',
    stateTeleoperating: 'Teleoperating',
    stateRecording: 'Recording',

    // Hardware
    hwReady: 'Hardware Ready',
    hwNotReady: 'Hardware Not Ready',
    hwUncalibrated: 'Uncalibrated',
    arms: 'Arms',
    cameras: 'Cameras',
    hwConnected: 'Connected',
    hwDisconnected: 'Disconnected',
    hwCalibrated: 'Calibrated',
    noArms: 'No arms detected',
    noCameras: 'No cameras detected',
    leader: 'Leader',
    follower: 'Follower',
    enablePreview: 'Enable Preview',
    disablePreview: 'Disable Preview',
    camerasDisabled: 'Camera preview disabled',
    connecting: 'Connecting...',
    startingTeleop: 'Starting teleop...',
    startingRecord: 'Starting recording...',
    hwInitializing: 'Initializing hardware, please wait',
    fillDatasetName: 'Please fill in dataset name',
    fillTaskDesc: 'Please fill in task description',
    servoPositions: 'Servo Positions',
    servoLoading: 'Reading servo data...',
    servoBusy: 'Serial port busy, will resume after teleop/recording stops',
  },
} as const

type TranslationKey = keyof typeof translations.zh

interface I18nStore {
  locale: Locale
  setLocale: (locale: Locale) => void
  t: (key: TranslationKey) => string
}

export const useI18n = create<I18nStore>((set, get) => ({
  locale: 'zh',
  setLocale: (locale) => set({ locale }),
  t: (key) => translations[get().locale][key] || key,
}))
