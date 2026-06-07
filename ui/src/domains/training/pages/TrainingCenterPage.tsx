import { useEffect, useRef, useState } from 'react'
import { useDatasetsStore } from '@/domains/datasets/store/useDatasetsStore'
import { useSessionStore } from '@/domains/session/store/useSessionStore'
import {
  cloudAutomationPolicy,
  useTrainingStore,
  type CloudDatasetSource,
  type CloudModelSource,
} from '@/domains/training/store/useTrainingStore'
import { useHubTransferStore } from '@/domains/hub/store/useHubTransferStore'
import { LossCurvePanel } from '@/domains/training/components/LossCurvePanel'
import { TrainingProgressPanel } from '@/domains/training/components/TrainingProgressPanel'
import { CloudIntentPanel } from '@/domains/training/components/CloudIntentPanel'
import { CloudSourcePanel } from '@/domains/training/components/CloudSourcePanel'
import { CloudProviderPanel } from '@/domains/training/components/CloudProviderPanel'
import { fetchProviderStatus } from '@/domains/provider/api/providerApi'
import { postJson } from '@/shared/api/client'
import { useI18n } from '@/i18n'

const POLICY_TYPES = [
  'act',
  'diffusion',
  'groot',
  'multi_task_dit',
  'pi0',
  'pi0_fast',
  'pi05',
  'reward_classifier',
  'sac',
  'sarm',
  'smolvla',
  'tdmpc',
  'vqbet',
  'wall_x',
  'xvla',
]

type CloudSourceKind =
  | 'official_reference'
  | 'platform_dataset'
  | 'public_reference'
  | 'user_object_storage'

type AccessMode = 'public' | 'saved_connection'
type ModelInitSource = 'auto' | 'external' | 'none'
type LaunchSummaryAction = 'intent' | 'data' | 'model' | 'resource'
type CloudQueuedTask = {
  id: string
  text: string
  createdAt: string
}

const CLOUD_TASK_QUEUE_KEY = 'evo_studio.training.task_queue'
const CLOUD_LAST_INTENT_KEY = 'evo_studio.training.last_intent'

function loadCloudTaskQueue(): CloudQueuedTask[] {
  if (typeof localStorage === 'undefined') return []
  try {
    const parsed = JSON.parse(localStorage.getItem(CLOUD_TASK_QUEUE_KEY) || '[]')
    if (!Array.isArray(parsed)) return []
    return parsed
      .map((item) => ({
        id: String(item?.id || ''),
        text: String(item?.text || '').trim(),
        createdAt: String(item?.createdAt || ''),
      }))
      .filter((item) => item.id && item.text)
      .slice(0, 20)
  } catch {
    return []
  }
}

function loadCloudLastIntent(): string {
  if (typeof localStorage === 'undefined') return ''
  return localStorage.getItem(CLOUD_LAST_INTENT_KEY) || ''
}

function SendArrowIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 12h13" />
      <path d="m13 6 6 6-6 6" />
    </svg>
  )
}

const CLOUD_SOURCE_OPTIONS: Array<{ id: CloudSourceKind; title: string; detail: string }> = [
  {
    id: 'official_reference',
    title: '官方开源数据集',
    detail: 'LIBERO 等官方公开数据集由平台后端自动下载/缓存，不需要用户提供云端 dataset id。',
  },
  {
    id: 'platform_dataset',
    title: 'Evo Studio 数据',
    detail: '我的 private 数据、已兑换 public 数据，训练时由云端读取。',
  },
  {
    id: 'public_reference',
    title: '公开数据链接',
    detail: '粘贴公开数据集链接，平台训练时按需读取，不要求用户理解存储协议。',
  },
  {
    id: 'user_object_storage',
    title: '我的云端数据',
    detail: '适合阿里云、实验室服务器、公司云盘等私有数据；平台只保存权限引用。',
  },
]

const DATASET_FORMATS = ['auto', 'lerobot', 'libero', 'robomimic_hdf5', 'webdataset', 'parquet', 'custom']

const OFFICIAL_DATASET_SOURCES: Record<string, { uri: string; format: string; label: string }> = {
  libero: {
    uri: 'hf://HuggingFaceVLA/libero',
    format: 'lerobot',
    label: 'HuggingFaceVLA/libero',
  },
}

const CHECKPOINT_FORMATS = [
  'auto',
  'lerobot_pretrained_model',
  'huggingface_transformers',
  'safetensors',
  'pytorch_bin',
  'lora_adapter',
  'robomimic_hdf5',
  'rlinf_artifact',
  'openpi_checkpoint',
  'custom',
]

const AUTODL_REGION_LABELS: Record<string, string> = {
  westDC2: '西北',
  westDC3: '西北',
  beijingDC1: '北京',
  beijingDC2: '北京',
  beijingDC3: 'V100专区',
  beijingDC4: 'L20专区',
  neimengDC1: '内蒙',
  neimengDC3: '内蒙',
  foshanDC1: '佛山',
  chongqingDC1: '重庆',
  yangzhouDC1: '3090专区',
}

function canonicalPolicyType(value: string): string {
  const key = value.trim().toLowerCase().replace(/_/g, '-')
  const aliases: Record<string, string> = {
    groot: 'gr00t',
    gr00tn1: 'gr00t',
    'gr00t-n1': 'gr00t',
    'pi0-fast': 'pi0',
    'pi0.5': 'pi05',
    'smol-vla': 'smolvla',
    rynn: 'rynnvla',
    'rynnvla-001': 'rynnvla',
    diffusion_policy: 'diffusion',
    'diffusion-policy': 'diffusion',
  }
  return aliases[key] || key
}

const BUILTIN_BENCHMARKS = [
  { id: 'libero', label: 'LIBERO' },
  { id: 'metaworld', label: 'MetaWorld' },
  { id: 'maniskill', label: 'ManiSkill' },
  { id: 'isaaclab', label: 'IsaacLab' },
  { id: 'custom', label: '其他 / 自定义' },
]

const TRAINING_WORKFLOWS = [
  {
    id: 'rlinf_vla',
    label: 'RLinf 后训练 / 评测',
    detail: '由平台把需求转换成 actor / rollout / env / runner / algorithm contract，适合 VLA + RL 后训练、评测和 smoke test。',
  },
  {
    id: 'vla_rl_backend',
    label: '通用项目执行',
    detail: '适合 OpenVLA-OFT、LeRobot、OpenPI、用户 Git 项目等已有 launcher；平台传入数据、模型、运行时和产物协议。',
  },
  {
    id: 'custom_project',
    label: '自定义项目',
    detail: '适合用户提供 Git 仓库、入口命令和产物协议的任务；仍应显式声明数据、模型、运行时和产物。',
  },
]

const RLINF_BACKEND_INTERFACE = {
  interfaceVersion: 'vla-rl-backend/v1',
  backendKind: 'rlinf',
  launchModes: ['project_backend', 'rlinf_frontend'],
  launcherKinds: ['python_module', 'python_script', 'deepspeed_script'],
  registryInjection: {
    field: 'rlinfExtModule',
    env: 'RLINF_EXT_MODULE',
  },
  algorithmToLauncherKind: {
    ppo: 'python_module',
    sac: 'python_module',
    grpo: 'python_module',
  },
  artifactContract: {
    contractFile: 'run_contract.json',
    requiredFields: ['backendKind', 'modelFamily', 'datasetPath', 'checkpointPath', 'artifactPath', 'metricPaths'],
    successMetricField: 'successMetric',
  },
}

const RLINF_RECIPE_CATALOG: Record<string, Record<string, unknown>> = {
  rlinf_vla_default: {
    recipeId: 'rlinf_vla_default',
    title: 'RLinf VLA 后训练 / 评测',
    framework: 'rlinf',
    workflow: 'rlinf_vla',
    runner: 'EmbodiedRunner',
    scheduler: 'Cluster',
    placementClass: 'HybridComponentPlacement',
    actorWorker: 'EmbodiedFSDPActor',
    rolloutWorker: 'MultiStepRolloutWorker',
    envWorker: 'EnvWorker',
    smokeFirst: true,
  },
}

function isObjectStorageUri(uri: string): boolean {
  return /^(s3|oss|r2|cos|minio):\/\//i.test(uri.trim())
}

function inferCloudSourceKindFromUri(uri: string): CloudSourceKind {
  const value = uri.trim()
  const lower = value.toLowerCase()
  if (!value) return 'official_reference'
  if (
    /^(hf|huggingface|modelscope|kaggle|dagshub):\/\//i.test(value)
    || /^(https?:\/\/)?(huggingface\.co|modelscope\.cn|www\.kaggle\.com|kaggle\.com|dagshub\.com)\//i.test(value)
  ) {
    return 'public_reference'
  }
  if (isObjectStorageUri(value) || lower.startsWith('http://') || lower.startsWith('https://') || lower.startsWith('/')) {
    return 'user_object_storage'
  }
  return 'public_reference'
}

function normalizePublicDatasetUri(uri: string): string {
  const value = uri.trim()
  if (!value) return ''
  if (/^(hf|huggingface):\/\//i.test(value)) {
    return value.replace(/^huggingface:\/\//i, 'hf://')
  }
  const match = value.match(/^https?:\/\/huggingface\.co\/(?:datasets\/)?([^/?#]+\/[^/?#]+)/i)
  if (match?.[1]) return `hf://${match[1]}`
  return value
}

function labeledFormat(format: string): string {
  return format === 'auto' ? '自动识别' : format
}

function inferDatasetFormatFromUri(uri: string): string {
  const value = uri.trim().toLowerCase()
  if (!value) return 'auto'
  if (value.endsWith('.parquet')) return 'parquet'
  if (value.endsWith('.hdf5') || value.endsWith('.h5')) return 'robomimic_hdf5'
  if (value.endsWith('.jsonl')) return 'lerobot'
  if (value.endsWith('.tar') || value.endsWith('.tar.gz') || value.endsWith('.tgz')) return 'webdataset'
  if (value.includes('libero')) return 'libero'
  if (value.includes('lerobot')) return 'lerobot'
  return 'auto'
}

function inferCheckpointFormatFromUri(uri: string, fallback = ''): string {
  const value = uri.trim().toLowerCase()
  if (!value) return fallback || 'auto'
  if (value.endsWith('.safetensors')) return 'safetensors'
  if (value.endsWith('.bin')) return 'pytorch_bin'
  if (value.endsWith('.pt') || value.endsWith('.pth') || value.endsWith('.ckpt')) return 'pytorch_bin'
  if (value.includes('openpi')) return 'openpi_checkpoint'
  if (value.includes('lerobot')) return 'lerobot_pretrained_model'
  if (value.includes('huggingface') || value.startsWith('hf://')) return 'huggingface_transformers'
  return fallback || 'auto'
}

function toRecord(value: unknown): Record<string, any> {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, any> : {}
}

function firstString(...values: unknown[]): string {
  for (const value of values) {
    if (typeof value === 'string' && value.trim()) return value.trim()
    if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  }
  return ''
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value.map(item => firstString(item)).filter(Boolean)
}

function looksLikeInternalPath(value: string): boolean {
  const text = value.trim()
  return text.startsWith('/root/')
    || text.startsWith('/workspace')
    || text.startsWith('/Users/')
    || text.startsWith('/private/tmp')
    || text.includes('/evo_studio/cache/')
}

function userFacingSourceValue(value: string, fallback: string): string {
  const text = value.trim()
  if (!text) return ''
  if (looksLikeInternalPath(text)) return fallback
  if (/^hf:\/\//i.test(text)) return text.replace(/^hf:\/\//i, '')
  const hfMatch = text.match(/^https?:\/\/huggingface\.co\/(?:datasets\/)?([^/?#]+\/[^/?#]+)/i)
  if (hfMatch?.[1]) return hfMatch[1]
  return text
}

function sanitizePlanText(value: unknown): string {
  const text = typeof value === 'string'
    ? value
    : JSON.stringify(value)
  return (text || '')
    .replace(/\/root\/autodl-tmp\/evo_studio\/cache\/datasets\/[^\s"',;，。；、)）\]}】]+/g, '云端数据缓存')
    .replace(/\/root\/autodl-tmp\/evo_studio\/cache\/models\/[^\s"',;，。；、)）\]}】]+/g, '云端模型缓存')
    .replace(/\/root\/autodl-tmp\/[^\s"',;，。；、)）\]}】]+/g, '云端项目目录')
    .replace(/\/workspace\/outputs[^\s"',;，。；、)）\]}】]*/g, '云端产物目录')
    .replace(/\/Users\/pearl\/[^\s"',;，。；、)）\]}】]+/g, '本地开发路径')
    .replace(/\/private\/tmp\/[^\s"',;，。；、)）\]}】]+/g, '本地临时配置')
}

function userFacingCloudRuntimeWarning(value: unknown): string {
  const text = firstString(value)
  const lower = text.toLowerCase()
  if (!text) return '后端需要重新绑定或检查当前云端实例。'
  if (lower.includes('configured ssh instance is not reachable') || lower.includes('error reading ssh protocol banner')) {
    return '当前云端实例没有返回 SSH 登录协议，请确认实例已开机，并重新绑定最新 SSH 命令。'
  }
  if (lower.includes('unable to reach evo_train') || lower.includes('connection refused')) {
    return 'EVO_Train 桥接服务暂时不可达，请先恢复本地桥接。'
  }
  return sanitizePlanText(text)
}

function extractVlaPlanPayload(data: unknown): Record<string, any> {
  const root = toRecord(data)
  const vlaPlan = toRecord(root.vlaPlan)
  if (Object.keys(vlaPlan).length > 0) return vlaPlan
  const plan = toRecord(root.plan)
  if (Object.keys(plan).length > 0) return plan
  return root
}

function isExistingSshRuntimePayload(data: unknown): boolean {
  const root = toRecord(data)
  const plan = extractVlaPlanPayload(root)
  const params = toRecord(plan.params)
  const runtime = toRecord(params.runtime)
  const runtimeMatch = toRecord(root.runtimeMatch)
  const configuration = toRecord(root.configuration)
  if (runtimeMatch.skipped === true) return true
  return [
    root.runtimeMode,
    runtimeMatch.runtimeMode,
    configuration.mode,
    configuration.deploymentMode,
    runtime.mode,
    runtime.runtimeMode,
  ].some(value => {
    const text = String(value || '').toLowerCase()
    return text === 'ssh' || text === 'ssh_existing_instance' || text === 'existing_ssh' || text === 'ssh_existing'
  })
}

function looksLikeAutodlRegionSign(value: string): boolean {
  return /^[a-z]+DC\d+$/iu.test(value.trim())
}

function autodlRegionLabel(regionSign: string): string {
  const keys = regionSign.split(',').map(item => item.trim()).filter(Boolean)
  if (keys.length > 1) {
    return Array.from(new Set(keys.map(key => AUTODL_REGION_LABELS[key] || key))).join(' / ')
  }
  const key = keys[0] || regionSign.trim()
  return AUTODL_REGION_LABELS[key] || key
}

function autodlRegionDisplayName(regionSign: string, rawName = ''): string {
  const cleanedRawName = rawName.trim()
  if (cleanedRawName && cleanedRawName !== regionSign && !looksLikeAutodlRegionSign(cleanedRawName)) {
    return cleanedRawName
  }
  return autodlRegionLabel(regionSign) || cleanedRawName || '未知地区'
}

function skuRegionSummary(sku: Record<string, unknown> | null | undefined): string {
  if (!sku) return ''
  const regionOptions = Array.isArray(sku.regionOptions) ? sku.regionOptions as Array<Record<string, unknown>> : []
  const labels = regionOptions
    .map((region) => autodlRegionDisplayName(
      firstString(region.regionSign, region.autodlRegionSign, region.region, region.dataCenter),
      firstString(region.regionName, region.dataCenterName),
    ))
    .filter(Boolean)
  const fallback = autodlRegionDisplayName(firstString(sku.autodlRegionSign, sku.regionSign, sku.region, sku.dataCenter))
  const uniqueLabels = Array.from(new Set(labels.length ? labels : fallback ? [fallback] : []))
  if (uniqueLabels.length <= 2) return uniqueLabels.join(' / ')
  return `${uniqueLabels[0]}等 ${uniqueLabels.length} 个地区`
}

function stripGpuCountSuffix(value: string): string {
  return value
    .replace(/\s*·\s*\d+\s*卡\s*$/u, '')
    .replace(/\s*x\s*\d+\s*$/u, '')
    .trim()
}

function skuRegionSigns(sku: Record<string, unknown> | null | undefined): string[] {
  if (!sku) return []
  const regionOptions = Array.isArray(sku.regionOptions) ? sku.regionOptions as Array<Record<string, unknown>> : []
  const signs = regionOptions
    .map(region => firstString(region.regionSign, region.autodlRegionSign, region.region, region.dataCenter))
    .filter(Boolean)
  const dataCenters = firstString(sku.autodlDataCenters)
    .split(',')
    .map(item => item.trim())
    .filter(Boolean)
  const fallback = firstString(sku.autodlRegionSign, sku.regionSign, sku.region, sku.dataCenter)
  return Array.from(new Set([...signs, ...dataCenters, fallback].filter(Boolean)))
}

function selectedRegionSigns(regionSign: string): string[] {
  return regionSign.split(',').map(item => item.trim()).filter(Boolean)
}

function skuAvailableInRegion(sku: Record<string, unknown> | null | undefined, regionSign: string): boolean {
  const selectedSigns = selectedRegionSigns(regionSign)
  if (selectedSigns.length === 0) return true
  const skuSigns = skuRegionSigns(sku)
  return selectedSigns.some(sign => skuSigns.includes(sign))
}

function skuRegionCapacity(sku: Record<string, unknown> | null | undefined, regionSign: string): { availableGpuCount: number; stockCount: number } {
  if (!sku) return { availableGpuCount: 0, stockCount: 0 }
  const selectedSigns = selectedRegionSigns(regionSign)
  if (selectedSigns.length > 1) {
    return selectedSigns.reduce((total, sign) => {
      const capacity = skuRegionCapacity(sku, sign)
      return {
        availableGpuCount: total.availableGpuCount + capacity.availableGpuCount,
        stockCount: total.stockCount + capacity.stockCount,
      }
    }, { availableGpuCount: 0, stockCount: 0 })
  }
  const selectedSign = selectedSigns[0] || ''
  const skuGpuCount = Math.max(1, Number(sku.gpuCount || 1) || 1)
  const fallbackAvailable = Number(sku.availableGpuCount || sku.stockCount || 0) || 0
  const fallbackStock = Number(sku.stockCount || 0) || (fallbackAvailable > 0 ? Math.floor(fallbackAvailable / skuGpuCount) : 0)
  if (!selectedSign) return { availableGpuCount: fallbackAvailable, stockCount: fallbackStock }
  const regionOptions = Array.isArray(sku.regionOptions) ? sku.regionOptions as Array<Record<string, unknown>> : []
  const region = regionOptions.find((item) => (
    firstString(item.regionSign, item.autodlRegionSign, item.region, item.dataCenter) === selectedSign
  ))
  if (!region) {
    return skuAvailableInRegion(sku, selectedSign)
      ? { availableGpuCount: fallbackAvailable, stockCount: fallbackStock }
      : { availableGpuCount: 0, stockCount: 0 }
  }
  const availableGpuCount = Number(region.availableGpuCount ?? region.idleGpuNum ?? region.gpuCount ?? 0) || 0
  const stockCount = Number(region.stockCount ?? 0) || (availableGpuCount > 0 ? Math.floor(availableGpuCount / skuGpuCount) : 0)
  return { availableGpuCount, stockCount }
}

function skuFamilyKey(sku: Record<string, unknown> | null | undefined): string {
  if (!sku) return ''
  const displayName = stripGpuCountSuffix(firstString(sku.displayName))
  const gpuName = stripGpuCountSuffix(firstString(sku.autodlGpuName, sku.gpuName))
  const gpuSpec = firstString(sku.gpuSpec)
  const skuId = firstString(sku.skuId).replace(/-\d+x$/u, '')
  return (gpuName || displayName || gpuSpec || skuId).toLowerCase()
}

function toCloudSourceKind(value: unknown): CloudSourceKind {
  const kind = firstString(value).toLowerCase()
  if (kind === 'public_reference' || kind === 'public_dataset' || kind === 'public_repo') return 'public_reference'
  if (kind === 'user_object_storage' || kind === 'object_storage' || kind === 'cloud_uri') return 'user_object_storage'
  return 'platform_dataset'
}

function nestedValue(source: Record<string, any>, path: string): unknown {
  return path.split('.').reduce<unknown>((current, key) => {
    if (!current || typeof current !== 'object') return undefined
    return (current as Record<string, unknown>)[key]
  }, source)
}

function firstValue(source: Record<string, any>, paths: string[]): unknown {
  for (const path of paths) {
    const value = nestedValue(source, path)
    if (value !== undefined && value !== null && value !== '') return value
  }
  return undefined
}

function formatParamValue(value: unknown): string {
  if (value === undefined || value === null || value === '') return ''
  if (Array.isArray(value)) return value.map(item => formatParamValue(item)).filter(Boolean).join('、')
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : String(value)
  if (typeof value === 'string') return value
  if (typeof value === 'object') {
    return Object.entries(value as Record<string, unknown>)
      .map(([key, item]) => {
        const text = formatParamValue(item)
        return text ? `${key}=${text}` : ''
      })
      .filter(Boolean)
      .join('；')
  }
  return String(value)
}

function parseManualParamValue(value: string): string | number | undefined {
  const text = value.trim()
  if (!text) return undefined
  const numeric = Number(text)
  return Number.isFinite(numeric) ? numeric : text
}

function humanizeRuntimeBlockingReason(reason: unknown): string {
  const text = String(reason || '').trim()
  if (!text) return ''
  const lower = text.toLowerCase()
  if (lower.includes('elastic deployment api is not enabled')) {
    return '当前部署只开启了 AutoDL 库存查询，还没有开启自动租机权限。需要团队账号开通 AutoDL 弹性部署 API 后再启动付费任务。'
  }
  if (lower.includes('sku is incomplete or disabled')) {
    return '这个机型来自实时库存发现，但当前部署还不能用它创建实例。'
  }
  return text
}

function inferBenchmarkKey(...values: unknown[]): string {
  const text = values
    .map(value => String(value || '').toLowerCase())
    .filter(Boolean)
    .join(' ')
  if (text.includes('maniskill')) return 'maniskill'
  if (text.includes('libero')) return 'libero'
  if (text.includes('metaworld')) return 'metaworld'
  if (text.includes('isaac')) return 'isaaclab'
  if (text.includes('robomimic')) return 'robomimic'
  return ''
}

function isUnresolvedModelText(value: string): boolean {
  const text = value.trim().toLowerCase()
  return !text || text === 'auto' || text === 'ai_resolved' || text === 'builtin_policy' || text === 'unknown'
}

function displayModelFamily(value: string, configName = ''): string {
  const text = `${value} ${configName}`.toLowerCase()
  if (text.includes('openvla') || text.includes('vla-oft')) return 'OpenVLA-OFT'
  if (text.includes('smolvla')) return 'SmolVLA'
  if (text.includes('tinyvla')) return 'TinyVLA'
  if (text.includes('rynnvla')) return 'RynnVLA'
  if (text.includes('pi0')) return 'pi0'
  if (text.includes('act')) return 'ACT'
  return value && value !== 'auto' ? value : '模型待解析'
}

function parseTrainStatusFromMessage(message: string): string {
  const statusMatch = message.match(/^status:\s*(.+)$/im)
  if (statusMatch?.[1]) return statusMatch[1].trim()
  const runningMatch = message.match(/^running:\s*(.+)$/im)
  if (runningMatch?.[1]?.trim().toLowerCase() === 'true') return 'Running'
  if (/failed|error:/i.test(message)) return 'Failed'
  if (/stopped/i.test(message)) return 'Stopped'
  if (/completed|success/i.test(message)) return 'Completed'
  return 'Unknown'
}

function summarizeTrainJobMessage(message: string): string {
  const text = message.trim()
  if (!text) return ''
  const status = parseTrainStatusFromMessage(text).toLowerCase()
  if (/ssh connection failed|error reading ssh protocol banner|configured ssh instance is not reachable/i.test(text)) {
    return '云端实例未连上，请重新绑定实例后继续。'
  }
  if (/tokenizers.*0\.22|tokenizers>=0\.19,<0\.20/i.test(text)) {
    return '任务暂停：OpenVLA-OFT 依赖版本不兼容，已记录到当前任务。'
  }
  if (/__EVO_GPU_UNAVAILABLE__|cloud_gpu_unavailable/i.test(text)) {
    return '任务暂停：当前运行环境没有检测到可用 GPU。'
  }
  if (/Evaluating Rollout Epochs:\s*100%|eval\/success_|eval\/num_trajectories|eval\/episode_len|success_rate/i.test(text)) {
    return '评测已执行，结果收集或任务判定需要修复。'
  }
  if (status.includes('failed')) return '任务暂停：需要处理后继续。'
  if (status.includes('running')) return '云端任务运行中。'
  const firstLine = text.split('\n').map(line => line.trim()).find(Boolean) || text
  return firstLine.length > 180 ? `${firstLine.slice(0, 180)}...` : firstLine
}

function isActiveCloudStatus(status: string): boolean {
  const text = status.toLowerCase()
  return ['running', 'submitting', 'submitted', 'pending', 'queued', 'starting', 'repairing'].some(item => text.includes(item))
}

export default function TrainingCenterPage() {
  const datasets = useDatasetsStore((state) => state.datasets)
  const loadDatasets = useDatasetsStore((state) => state.loadDatasets)
  const session = useSessionStore((state) => state.session)
  const policies = useTrainingStore((state) => state.policies)
  const loadPolicies = useTrainingStore((state) => state.loadPolicies)
  const loadCloudBridgeStatus = useTrainingStore((state) => state.loadCloudBridgeStatus)
  const loadCloudResources = useTrainingStore((state) => state.loadCloudResources)
  const loadRuntimeMatch = useTrainingStore((state) => state.loadRuntimeMatch)
  const loadSourcePreflight = useTrainingStore((state) => state.loadSourcePreflight)
  const loadAuthConnections = useTrainingStore((state) => state.loadAuthConnections)
  const saveAuthConnection = useTrainingStore((state) => state.saveAuthConnection)
  const restoreCurrentTrainJob = useTrainingStore((state) => state.restoreCurrentTrainJob)
  const detachCurrentTrainJob = useTrainingStore((state) => state.detachCurrentTrainJob)
  const doTrainStart = useTrainingStore((state) => state.doTrainStart)
  const doCloudTrainPlan = useTrainingStore((state) => state.doCloudTrainPlan)
  const clearCloudTrainPlan = useTrainingStore((state) => state.clearCloudTrainPlan)
  const cancelCloudChecks = useTrainingStore((state) => state.cancelCloudChecks)
  const cancelCloudStartRequest = useTrainingStore((state) => state.cancelCloudStartRequest)
  const doCloudTrainStart = useTrainingStore((state) => state.doCloudTrainStart)
  const repairCloudTraining = useTrainingStore((state) => state.repairCloudTraining)
  const doTrainStop = useTrainingStore((state) => state.doTrainStop)
  const cloudAutomationMode = useTrainingStore((state) => state.cloudAutomationMode)
  const currentTrainJobId = useTrainingStore((state) => state.currentTrainJobId)
  const currentTrainMode = useTrainingStore((state) => state.currentTrainMode)
  const restartableCloudJobId = useTrainingStore((state) => state.restartableCloudJobId)
  const trainJobHistory = useTrainingStore((state) => state.trainJobHistory)
  const trainJobMessage = useTrainingStore((state) => state.trainJobMessage)
  const trainPlanMessage = useTrainingStore((state) => state.trainPlanMessage)
  const trainPlan = useTrainingStore((state) => state.trainPlan)
  const cloudBridgeStatus = useTrainingStore((state) => state.cloudBridgeStatus)
  const cloudResourceCatalog = useTrainingStore((state) => state.cloudResourceCatalog)
  const cloudRuntimeMatch = useTrainingStore((state) => state.cloudRuntimeMatch)
  const cloudSourcePreflight = useTrainingStore((state) => state.cloudSourcePreflight)
  const authConnections = useTrainingStore((state) => state.authConnections)
  const trainingLoading = useTrainingStore((state) => state.trainingLoading)
  const trainingPlanLoading = useTrainingStore((state) => state.trainingPlanLoading)
  const runtimeMatchLoading = useTrainingStore((state) => state.runtimeMatchLoading)
  const sourcePreflightLoading = useTrainingStore((state) => state.sourcePreflightLoading)
  const trainingStopLoading = useTrainingStore((state) => state.trainingStopLoading)
  const hubLoading = useHubTransferStore((state) => state.hubLoading)
  const hubProgress = useHubTransferStore((state) => state.hubProgress)
  const pushPolicy = useHubTransferStore((state) => state.pushPolicy)
  const pullPolicy = useHubTransferStore((state) => state.pullPolicy)
  const { t } = useI18n()
  const runtimeDatasets = datasets.filter((dataset) => dataset.capabilities.can_train && dataset.runtime)
  const trainableDatasets = datasets.filter((dataset) => dataset.capabilities.can_train)

  const [trainDataset, setTrainDataset] = useState('')
  const [trainMode, setTrainMode] = useState<'local' | 'cloud'>('cloud')
  const [policyType, setPolicyType] = useState('act')
  const [trainSteps, setTrainSteps] = useState(100000)
  const [trainDevice, setTrainDevice] = useState('cuda')
  const [cloudLearningRate, setCloudLearningRate] = useState('')
  const [cloudBatchSize, setCloudBatchSize] = useState('')
  const [cloudEpochs, setCloudEpochs] = useState('')
  const [cloudWarmupSteps, setCloudWarmupSteps] = useState('')
  const [cloudGradientAccumulationSteps, setCloudGradientAccumulationSteps] = useState('')
  const [cloudLoraRank, setCloudLoraRank] = useState('')
  const [pullPolicyRepo, setPullPolicyRepo] = useState('')
  const [cloudUsername, setCloudUsername] = useState(() => {
    if (typeof localStorage === 'undefined') return 'pearl'
    return localStorage.getItem('roboclaw.dataset.username') || 'pearl'
  })
  const [cloudWorkflow, setCloudWorkflow] = useState('rlinf_vla')
  const [rlinfAlgorithm, setRlinfAlgorithm] = useState('auto')
  const [rlinfPlacementStrategy, setRlinfPlacementStrategy] = useState('single_node')
  const [rlinfRolloutBackend, setRlinfRolloutBackend] = useState('huggingface')
  const [rlinfGroupSize, setRlinfGroupSize] = useState(1)
  const [cloudProvider, setCloudProvider] = useState<'autodl' | 'aliyun'>('autodl')
  const [cloudRegionSign, setCloudRegionSign] = useState('')
  const [cloudSkuId, setCloudSkuId] = useState('')
  const [cloudImageId, setCloudImageId] = useState('')
  const [cloudGpuCount, setCloudGpuCount] = useState(1)
  const [cloudReplicaCount, setCloudReplicaCount] = useState(1)
  const [cloudSourceKind, setCloudSourceKind] = useState<CloudSourceKind>('official_reference')
  const [cloudDatasetId, setCloudDatasetId] = useState('')
  const [cloudSourceUri, setCloudSourceUri] = useState('')
  const [cloudSourceConfirmedKey, setCloudSourceConfirmedKey] = useState('')
  const [cloudSourceConfirmedAt, setCloudSourceConfirmedAt] = useState('')
  const [cloudDataAccessMode, setCloudDataAccessMode] = useState<AccessMode>('public')
  const [cloudAuthRef, setCloudAuthRef] = useState('')
  const [cloudFormat, setCloudFormat] = useState('auto')
  const [cloudEnvironmentMode, setCloudEnvironmentMode] = useState<'none' | 'benchmark'>('none')
  const [cloudBenchmark, setCloudBenchmark] = useState('libero')
  const [cloudEnvironmentHint, setCloudEnvironmentHint] = useState('')
  const [cloudModelUri, setCloudModelUri] = useState('')
  const [cloudModelInitSource, setCloudModelInitSource] = useState<ModelInitSource>('auto')
  const [cloudModelAccessMode, setCloudModelAccessMode] = useState<AccessMode>('public')
  const [cloudModelAuthRef, setCloudModelAuthRef] = useState('')
  const [cloudCheckpointFormat, setCloudCheckpointFormat] = useState('auto')
  const [cloudIntent, setCloudIntent] = useState(loadCloudLastIntent)
  const [cloudTaskQueue, setCloudTaskQueue] = useState<CloudQueuedTask[]>(loadCloudTaskQueue)
  const [cloudExecutionNotes, setCloudExecutionNotes] = useState('')
  const [cloudExecutionMode, setCloudExecutionMode] = useState<'auto' | 'prepare' | 'gpu'>('auto')
  const [showAdvancedCloudOptions, setShowAdvancedCloudOptions] = useState(false)
  const [showExpertCloudOptions, setShowExpertCloudOptions] = useState(false)
  const [inlineConfirmEdit, setInlineConfirmEdit] = useState<LaunchSummaryAction | ''>('')
  const [showFormatOverrides, setShowFormatOverrides] = useState(false)
  const [aiConfiguredParams, setAiConfiguredParams] = useState<Record<string, unknown> | null>(null)
  const [lastAppliedPlanKey, setLastAppliedPlanKey] = useState('')
  const [showAuthConnectionForm, setShowAuthConnectionForm] = useState(false)
  const [authConnectionKind, setAuthConnectionKind] = useState<'data' | 'model'>('model')
  const [authConnectionProvider, setAuthConnectionProvider] = useState('huggingface')
  const [authConnectionId, setAuthConnectionId] = useState('')
  const [authConnectionLabel, setAuthConnectionLabel] = useState('')
  const [authConnectionToken, setAuthConnectionToken] = useState('')
  const [authConnectionAccessKey, setAuthConnectionAccessKey] = useState('')
  const [authConnectionSecretKey, setAuthConnectionSecretKey] = useState('')
  const [authConnectionMessage, setAuthConnectionMessage] = useState('')
  const [sshBindCommand, setSshBindCommand] = useState('')
  const [sshBindPassword, setSshBindPassword] = useState('')
  const [sshBindKeyPath, setSshBindKeyPath] = useState('')
  const [sshBindLoading, setSshBindLoading] = useState(false)
  const [sshBindMessage, setSshBindMessage] = useState('')
  const [showSshRuntimeBind, setShowSshRuntimeBind] = useState(false)
  const [aiProviderConfigured, setAiProviderConfigured] = useState<boolean | null>(null)
  const [, setAiProviderLabel] = useState('')
  const [rlinfCatalogSummary, setRlinfCatalogSummary] = useState<{ configured: boolean; count: number; benchmarks: string[] }>({
    configured: false,
    count: 0,
    benchmarks: [],
  })
  const advancedCloudOptionsRef = useRef<HTMLDivElement | null>(null)
  const cloudIntentRef = useRef<HTMLTextAreaElement | null>(null)
  const cloudStartActivationRef = useRef(0)
  const queuedPlanningRef = useRef('')
  const cloudSourceUriRef = useRef<HTMLInputElement | null>(null)
  const cloudModelUriRef = useRef<HTMLInputElement | null>(null)
  const sshBindSectionRef = useRef<any>(null)
  const cloudCheckLoading = trainingPlanLoading || runtimeMatchLoading || sourcePreflightLoading

  const invalidateCloudPlan = () => {
    clearCloudTrainPlan()
    setAiConfiguredParams(null)
  }

  const enqueueCloudTask = (text: string) => {
    const trimmed = text.trim()
    if (!trimmed) return
    const item: CloudQueuedTask = {
      id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
      text: trimmed,
      createdAt: new Date().toISOString(),
    }
    setCloudTaskQueue((current) => [...current, item].slice(-20))
    setCloudIntent('')
    invalidateCloudPlan()
  }

  const removeQueuedCloudTask = (id: string) => {
    setCloudTaskQueue((current) => current.filter((item) => item.id !== id))
  }

  useEffect(() => {
    void loadDatasets()
    void loadPolicies()
    void loadCloudBridgeStatus()
    void loadCloudResources(cloudProvider)
    void loadAuthConnections(cloudUsername)
    void restoreCurrentTrainJob()
  }, [loadDatasets, loadPolicies, loadCloudBridgeStatus, loadCloudResources, loadAuthConnections, restoreCurrentTrainJob, cloudUsername, cloudProvider])

  useEffect(() => {
    if (typeof localStorage === 'undefined') return
    localStorage.setItem(CLOUD_TASK_QUEUE_KEY, JSON.stringify(cloudTaskQueue))
  }, [cloudTaskQueue])

  useEffect(() => {
    if (typeof localStorage === 'undefined') return
    const trimmed = cloudIntent.trim()
    if (trimmed) {
      localStorage.setItem(CLOUD_LAST_INTENT_KEY, cloudIntent)
    } else {
      localStorage.removeItem(CLOUD_LAST_INTENT_KEY)
    }
  }, [cloudIntent])

  useEffect(() => {
    void loadCloudResources(cloudProvider)
    setCloudRegionSign('')
    setCloudSkuId('')
    setCloudImageId('')
  }, [cloudProvider, loadCloudResources])

  useEffect(() => {
    if (cloudBridgeStatus?.enabled === false) return
    const refresh = () => {
      void loadCloudResources(cloudProvider)
    }
    const timer = window.setInterval(refresh, 10000)
    const refreshWhenVisible = () => {
      if (!document.hidden) refresh()
    }
    document.addEventListener('visibilitychange', refreshWhenVisible)
    return () => {
      window.clearInterval(timer)
      document.removeEventListener('visibilitychange', refreshWhenVisible)
    }
  }, [cloudBridgeStatus?.enabled, cloudProvider, loadCloudResources])

  useEffect(() => {
    let cancelled = false
    async function loadProviderStatus() {
      try {
        const payload = await fetchProviderStatus()
        if (cancelled) return
        setAiProviderConfigured(Boolean(payload.active_provider_configured))
        const active = payload.providers.find(provider => provider.name === payload.active_provider)
        setAiProviderLabel(active?.label || payload.active_provider || '')
      } catch {
        if (!cancelled) {
          setAiProviderConfigured(false)
          setAiProviderLabel('')
        }
      }
    }
    void loadProviderStatus()
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    let cancelled = false
    async function loadRlinfCatalog() {
      try {
        const response = await fetch('/api/vla-rl/rlinf-catalog')
        const payload = await response.json()
        if (cancelled) return
        const configs = Array.isArray(payload.configs) ? payload.configs : []
        const benchmarks = Array.from(new Set<string>(
          configs
            .map((item: any) => String(item.benchmark || '').trim())
            .filter(Boolean),
        )).slice(0, 4)
        setRlinfCatalogSummary({
          configured: Boolean(payload.configured),
          count: Number(payload.count || configs.length || 0),
          benchmarks,
        })
      } catch {
        if (!cancelled) setRlinfCatalogSummary({ configured: false, count: 0, benchmarks: [] })
      }
    }
    void loadRlinfCatalog()
    return () => { cancelled = true }
  }, [])

  const promptPushPolicy = (value: string) => {
    const repoId = prompt(t('enterRepoId'))
    if (!repoId) return
    void pushPolicy(value, repoId)
  }

  const selectedCloudDataset = trainableDatasets.find(dataset => dataset.id === cloudDatasetId)
  const dataAuthConnections = authConnections.filter(connection => connection.kind === 'data' || connection.kind === 'both')
  const modelAuthConnections = authConnections.filter(connection => connection.kind === 'model' || connection.kind === 'both')
  const objectStorageAuthProviders = new Set(['s3', 'oss', 'r2', 'cos', 'minio'])
  const usesObjectStorageSecret = objectStorageAuthProviders.has(authConnectionProvider)
  const officialDatasetSource = OFFICIAL_DATASET_SOURCES[cloudBenchmark]
  const cloudDatasetUri = cloudSourceKind === 'official_reference'
    ? officialDatasetSource?.uri || ''
    : selectedCloudDataset?.source_dataset
    || selectedCloudDataset?.runtime?.repo_id
    || ''
  const trimmedCloudUri = cloudSourceUri.trim()
  const trimmedCloudModelUri = cloudModelUri.trim()
  const effectiveDatasetUri = cloudSourceKind === 'platform_dataset' || cloudSourceKind === 'official_reference' ? cloudDatasetUri : trimmedCloudUri
  const normalizedPublicDatasetUri = cloudSourceKind === 'public_reference' ? normalizePublicDatasetUri(trimmedCloudUri) : ''
  const inferredDatasetFormat = cloudFormat === 'auto'
    ? (cloudSourceKind === 'official_reference' ? officialDatasetSource?.format || 'auto' : inferDatasetFormatFromUri(effectiveDatasetUri))
    : cloudFormat
  const publicSourceConfirmationKey = cloudSourceKind === 'public_reference'
    ? `${cloudProvider}|${normalizedPublicDatasetUri}|${inferredDatasetFormat}`
    : ''
  const rawAiModelFamily = canonicalPolicyType(firstString(
    aiConfiguredParams?.modelFamily,
    aiConfiguredParams?.policyType,
    aiConfiguredParams?.model,
  ))
  const aiModelSourceForFamily = toRecord(aiConfiguredParams?.modelSource)
  const aiModelSourceTypeForFamily = firstString(
    aiModelSourceForFamily.sourceType,
    aiModelSourceForFamily.type,
  ).toLowerCase()
  const aiModelHasConcreteSource = Boolean(firstString(
    aiModelSourceForFamily.uri,
    aiModelSourceForFamily.modelUri,
    aiModelSourceForFamily.repoId,
    aiModelSourceForFamily.repository,
    aiModelSourceForFamily.checkpoint,
    aiModelSourceForFamily.checkpointName,
  ))
  const aiModelFamily = aiModelSourceTypeForFamily === 'ai_resolved' && !aiModelHasConcreteSource
    ? ''
    : rawAiModelFamily
  const effectiveCloudPolicyType = aiModelFamily || (cloudModelInitSource === 'auto' ? 'auto' : policyType)
  const inferredCheckpointFormat = cloudCheckpointFormat === 'auto'
    ? inferCheckpointFormatFromUri(cloudModelUri, '')
    : cloudCheckpointFormat
  const cloudSkus = cloudResourceCatalog?.skus || []
  const regionRowsBySign = Array.from(cloudSkus.reduce((map, sku) => {
    const regionOptions = Array.isArray(sku.regionOptions) ? sku.regionOptions as Array<Record<string, unknown>> : []
    for (const region of regionOptions) {
      const regionSign = firstString(region.regionSign, region.autodlRegionSign, region.region, region.dataCenter)
      if (!regionSign) continue
      const rawName = firstString(region.regionName, region.dataCenterName, region.dataCenter)
      const label = autodlRegionDisplayName(regionSign, rawName)
      const availableGpuCount = Number(region.availableGpuCount ?? region.idleGpuNum ?? region.stockCount ?? 0) || 0
      const existing = map.get(regionSign)
      if (!existing || availableGpuCount > existing.availableGpuCount) {
        map.set(regionSign, { regionSign, label, availableGpuCount })
      }
    }
    for (const regionSign of skuRegionSigns(sku)) {
      if (!regionSign || map.has(regionSign)) continue
      map.set(regionSign, {
        regionSign,
        label: autodlRegionDisplayName(regionSign),
        availableGpuCount: Number(sku.availableGpuCount || sku.stockCount || 0) || 0,
      })
    }
    return map
  }, new Map<string, { regionSign: string; label: string; availableGpuCount: number }>()).values())
  const regionCatalogRows = Array.from(regionRowsBySign.reduce((map, row) => {
    const key = row.label || row.regionSign
    const existing = map.get(key)
    if (existing) {
      existing.regionSign = Array.from(new Set([
        ...existing.regionSign.split(',').map(item => item.trim()).filter(Boolean),
        row.regionSign,
      ])).join(',')
      existing.availableGpuCount += row.availableGpuCount
    } else {
      map.set(key, { ...row })
    }
    return map
  }, new Map<string, { regionSign: string; label: string; availableGpuCount: number }>()).values())
    .sort((a, b) => b.availableGpuCount - a.availableGpuCount)
  const regionFilteredCloudSkus = cloudRegionSign
    ? cloudSkus.filter(sku => skuAvailableInRegion(sku, cloudRegionSign))
    : cloudSkus
  const machineOptions = Array.from(regionFilteredCloudSkus.reduce((map, sku) => {
    const key = skuFamilyKey(sku)
    if (!key) return map
    const capacity = skuRegionCapacity(sku, cloudRegionSign)
    const gpuCount = Math.max(1, Number(sku.gpuCount || 1) || 1)
    const hourly = Number(sku.hourlyPriceCents ?? sku.costHourlyCents ?? 0) || 0
    const perGpuHourly = hourly > 0 ? Math.round(hourly / gpuCount) : 0
    const displayName = stripGpuCountSuffix(firstString(sku.autodlGpuName, sku.gpuName, sku.displayName, sku.gpuSpec, sku.skuId))
    const existing = map.get(key)
    if (!existing) {
      map.set(key, {
        key,
        label: displayName || key,
        sku,
        availableGpuCount: capacity.availableGpuCount,
        perGpuHourlyCents: perGpuHourly,
        maxGpuCountPerInstance: Number(sku.maxGpuCountPerInstance || gpuCount || 1) || 1,
        hasReadySku: sku.readyToStart !== false,
        stockStatus: sku.stockStatus || '',
      })
      return map
    }
    if (capacity.availableGpuCount > existing.availableGpuCount) {
      existing.availableGpuCount = capacity.availableGpuCount
    }
    if ((!existing.perGpuHourlyCents || (perGpuHourly > 0 && perGpuHourly < existing.perGpuHourlyCents))) {
      existing.perGpuHourlyCents = perGpuHourly
    }
    existing.maxGpuCountPerInstance = Math.max(
      existing.maxGpuCountPerInstance,
      Number(sku.maxGpuCountPerInstance || gpuCount || 1) || 1,
    )
    existing.hasReadySku = existing.hasReadySku || sku.readyToStart !== false
    if (existing.stockStatus === 'sold_out' && sku.stockStatus !== 'sold_out') {
      existing.stockStatus = sku.stockStatus || ''
    }
    if (gpuCount === Math.max(1, cloudGpuCount)) {
      existing.sku = sku
    }
    return map
  }, new Map<string, {
    key: string
    label: string
    sku: Record<string, unknown>
    availableGpuCount: number
    perGpuHourlyCents: number
    maxGpuCountPerInstance: number
    hasReadySku: boolean
    stockStatus: unknown
  }>()).values())
    .sort((a, b) => b.availableGpuCount - a.availableGpuCount)
  const selectedCloudSkuRaw = regionFilteredCloudSkus.find(sku => String(sku.skuId || '') === cloudSkuId)
    || cloudSkus.find(sku => String(sku.skuId || '') === cloudSkuId)
    || null
  const selectedSkuFamilyKey = skuFamilyKey(selectedCloudSkuRaw)
  const selectedMachineOption = machineOptions.find(item => item.key === selectedSkuFamilyKey) || null
  const selectedMachineLabel = selectedMachineOption?.label
    || stripGpuCountSuffix(firstString(
      selectedCloudSkuRaw?.autodlGpuName,
      selectedCloudSkuRaw?.gpuName,
      selectedCloudSkuRaw?.displayName,
      selectedCloudSkuRaw?.gpuSpec,
      selectedCloudSkuRaw?.skuId,
    ))
  const selectedCloudSku = selectedCloudSkuRaw && selectedSkuFamilyKey
    ? regionFilteredCloudSkus.find(sku => (
      skuFamilyKey(sku) === selectedSkuFamilyKey
      && Number(sku.gpuCount || 1) === cloudGpuCount
      && sku.stockStatus !== 'sold_out'
      && skuAvailableInRegion(sku, cloudRegionSign)
    )) || selectedCloudSkuRaw
    : selectedCloudSkuRaw
  const effectiveCloudSkuId = String(selectedCloudSku?.skuId || cloudSkuId || '')
  const selectedCloudImage = (cloudResourceCatalog?.images || []).find(image => String(image.imageId || '') === cloudImageId) || null
  const cloudDeploymentMode = String(cloudBridgeStatus?.deploymentMode || '').toLowerCase()
  const isManagedComputePool = cloudDeploymentMode === 'managed'
  const isExistingDebugInstance = cloudDeploymentMode === 'ssh'
  const isExistingSshRuntime = isExistingDebugInstance || isExistingSshRuntimePayload(trainPlan)
  const cloudBridgeChecking = !cloudBridgeStatus
  const cloudBridgeConnected = cloudBridgeStatus?.enabled === true
  const cloudRuntimeReady = cloudBridgeConnected && cloudBridgeStatus?.configurationReady !== false
  const cloudGpuReady = cloudBridgeStatus?.gpuReady !== false
  const autoPrepareOnNoGpu = cloudExecutionMode === 'auto' && isExistingDebugInstance && cloudRuntimeReady && !cloudGpuReady
  const cloudCatalogSummary = cloudBridgeStatus?.resourceCatalog || {}
  const readySkuCount = Number(cloudCatalogSummary.readySkuCount ?? 0)
  const totalSkuCount = Number(cloudCatalogSummary.skuCount ?? 0)
  const readyImageCount = Number(cloudCatalogSummary.readyImageCount ?? 0)
  const totalImageCount = Number(cloudCatalogSummary.imageCount ?? 0)
  const canShowInternalTrainingDebug = typeof window !== 'undefined'
    && new URLSearchParams(window.location.search).has('debugTraining')
  const canRebindSshRuntime = canShowInternalTrainingDebug
    || isExistingDebugInstance
    || (!cloudBridgeChecking && (!cloudBridgeConnected || !cloudRuntimeReady))
  const selectedHourlyCostCents = Number(
    selectedCloudSku?.hourlyPriceCents
    ?? selectedCloudSku?.costHourlyCents
    ?? 0,
  )
  const selectedSkuStockStatus = String(selectedCloudSku?.stockStatus || '').toLowerCase()
  const selectedSkuSoldOut = selectedSkuStockStatus === 'sold_out'
  const selectedSkuGpuCount = Number(selectedCloudSku?.gpuCount || 1)
  const selectedRegionCapacity = skuRegionCapacity(selectedCloudSku, cloudRegionSign)
  const selectedStockCount = selectedRegionCapacity.stockCount
  const selectedAvailableGpuCount = selectedRegionCapacity.availableGpuCount
  const selectedRegionOptions = Array.isArray(selectedCloudSku?.regionOptions) ? selectedCloudSku.regionOptions as Array<Record<string, unknown>> : []
  const selectedRecommendedRegion = String(selectedCloudSku?.autodlRegionSign || selectedRegionOptions[0]?.regionSign || '')
  const selectedRegionCount = selectedRegionOptions.length
  const selectedRegionRows = selectedRegionOptions
    .map((region) => {
      const regionSign = firstString(region.regionSign, region.autodlRegionSign, region.region, region.dataCenter)
      const rawName = firstString(region.regionName, region.dataCenterName, region.dataCenter)
      const availableGpuCount = Number(region.availableGpuCount ?? region.idleGpuNum ?? region.stockCount ?? 0)
      return {
        regionSign,
        label: autodlRegionDisplayName(regionSign, rawName),
        availableGpuCount,
      }
    })
    .filter(region => selectedRegionSigns(cloudRegionSign).length === 0 || selectedRegionSigns(cloudRegionSign).includes(region.regionSign))
    .filter(region => region.regionSign || region.label || region.availableGpuCount > 0)
    .sort((a, b) => b.availableGpuCount - a.availableGpuCount)
  const selectedMaxGpuCountPerInstance = 12
  const gpuCountOptions = Array.from({ length: selectedMaxGpuCountPerInstance }, (_, index) => index + 1)
  const maxReplicaCountByStock = selectedAvailableGpuCount > 0
    ? Math.max(1, Math.floor(selectedAvailableGpuCount / Math.max(1, cloudGpuCount)))
    : 8
  const replicaCountOptions = Array.from(
    { length: Math.max(1, Math.min(32, maxReplicaCountByStock)) },
    (_, index) => index + 1,
  )
  const requestedGpuTotal = Math.max(1, cloudGpuCount) * Math.max(1, cloudReplicaCount)
  const selectedSkuTooSmallForContainer = Boolean(
    selectedCloudSku
    && selectedCloudSku.source !== 'autodl_api_discovered'
    && !selectedCloudSku.maxGpuCountPerInstance
    && selectedSkuGpuCount > 0
    && selectedSkuGpuCount < cloudGpuCount,
  )
  const resourceRequestTooLarge = Boolean(
    selectedCloudSku
    && (
      selectedSkuTooSmallForContainer
      || (selectedStockCount > 0 && cloudReplicaCount > selectedStockCount)
      || (selectedAvailableGpuCount > 0 && requestedGpuTotal > selectedAvailableGpuCount)
    ),
  )
  const effectiveCloudImageId = cloudImageId
  const runtimeCandidate = cloudRuntimeMatch?.matches?.find(candidate => {
    const candidateSku = firstString(candidate.skuId, toRecord(candidate.sku).skuId)
    const candidateImage = firstString(candidate.imageId, toRecord(candidate.image).imageId)
    const skuOk = !effectiveCloudSkuId || !candidateSku || candidateSku === effectiveCloudSkuId
    const imageOk = !effectiveCloudImageId || !candidateImage || candidateImage === effectiveCloudImageId
    return skuOk && imageOk
  }) || cloudRuntimeMatch?.matches?.[0] || null
  const runtimeCandidateSkuId = runtimeCandidate ? firstString(runtimeCandidate.skuId, toRecord(runtimeCandidate.sku).skuId) : ''
  const runtimeCandidateImageId = runtimeCandidate ? firstString(runtimeCandidate.imageId, toRecord(runtimeCandidate.image).imageId) : ''
  const runtimeBlockingSource = Array.isArray(runtimeCandidate?.blocking)
    ? runtimeCandidate?.blocking
    : (Array.isArray((runtimeCandidate as any)?.blockingReasons) ? (runtimeCandidate as any).blockingReasons : [])
  const runtimeBlocking = runtimeBlockingSource.filter(Boolean) as unknown[]
  const runtimeBlockingMessages = runtimeBlocking
    .map((reason: unknown) => humanizeRuntimeBlockingReason(reason))
    .filter(Boolean)
  const runtimeBlockedByAutodlDeploymentPermission = !isExistingSshRuntime && runtimeBlocking.some((reason: unknown) => (
    String(reason).toLowerCase().includes('elastic deployment api is not enabled')
  ))
  const runtimeIsIncompatible = !isExistingSshRuntime && Boolean(runtimeCandidate && runtimeCandidate.compatible === false)
  const runtimeRisks = Array.isArray((runtimeCandidate as any)?.risks)
    ? (runtimeCandidate as any).risks.map((item: unknown) => String(item)).filter(Boolean)
    : []
  const runtimeReadyLabel = runtimeMatchLoading
    ? '正在检查资源兼容性...'
    : runtimeCandidate
      ? (
          runtimeBlockedByAutodlDeploymentPermission
            ? '当前可查看库存，但暂不能自动租机'
            : (runtimeIsIncompatible ? '当前资源组合不可用' : (runtimeRisks.length > 0 ? '当前资源组合可尝试，需启动后健康检查' : '当前资源组合可用'))
        )
      : '等待资源兼容性检查'
  const sourcePreflightSource = toRecord(cloudSourcePreflight?.source)
  const sourcePreflightSize = firstString(sourcePreflightSource.estimatedSize) || '未知'
  const sourcePreflightPath = firstString(sourcePreflightSource.resolvedPath)
  const sourcePreflightRisks = Array.isArray(sourcePreflightSource.risks)
    ? sourcePreflightSource.risks.map(item => String(item)).filter(Boolean)
    : []
  const sourcePreflightWarnings = Array.isArray(sourcePreflightSource.warnings)
    ? sourcePreflightSource.warnings.map(item => String(item)).filter(Boolean)
    : []
  const canStartTraining = (session.state === 'idle' || session.state === 'error') && !trainingLoading
  const displayedVlaPlan = trainPlan ? extractVlaPlanPayload(trainPlan) : null
  const hasCloudPlan = Boolean(trainPlan && displayedVlaPlan && Object.keys(displayedVlaPlan).length > 0)
  const displayedPlanner = toRecord(displayedVlaPlan?.planner || toRecord(trainPlan || {}).aiPlan)
  const displayedPlannerSource = firstString(displayedPlanner.source)
  const displayedPlannerModel = firstString(displayedPlanner.providerModel, displayedPlanner.model)
  const aiPlannerWarnings = [
    ...stringArray(displayedVlaPlan?.warnings),
    ...stringArray(toRecord(trainPlan || {}).aiPlan?.warnings),
  ]
  const aiPlannerWarningText = aiPlannerWarnings.join('；')
  const aiPlannerUnavailable = Boolean(displayedPlannerSource && displayedPlannerSource !== 'llm')
  const clarifyingQuestions = Array.isArray(displayedVlaPlan?.clarifyingQuestions)
    ? displayedVlaPlan.clarifyingQuestions.map((item: unknown) => String(item).trim()).filter(Boolean)
    : []
  const intentUnderstanding = toRecord(displayedVlaPlan?.intentUnderstanding)
  const intentUnderstandingSummary = firstString(
    intentUnderstanding.objective,
    intentUnderstanding.summary,
    intentUnderstanding.goal,
    displayedVlaPlan?.aiSummary,
  )
  const aiPlannerUnavailableLabel = displayedPlannerSource === 'llm_unconfigured'
    ? '未接入大模型'
    : displayedPlannerSource === 'llm_timeout'
      ? '大模型规划超时'
      : displayedPlannerSource === 'llm_error'
        ? '大模型调用失败'
        : displayedPlannerSource === 'llm_parse_error'
          ? '大模型输出无法解析'
          : '未完成大模型规划'
  const aiPlannerUnavailableText = displayedPlannerSource === 'llm_unconfigured'
    ? '还没有可用的 AI provider。先去设置页保存 API key 和 base URL。'
    : displayedPlannerSource === 'llm_timeout'
      ? '已接入大模型，但这次中转 API 没有在时限内返回。可以直接重试；连续失败请检查 API base、key、模型名和中转余额/限流。'
      : displayedPlannerSource === 'llm_parse_error'
        ? 'AI 返回了内容，但不是平台可执行的方案。可以重试一次。'
        : displayedPlannerSource === 'llm_error'
          ? `已接入大模型，但调用失败。${aiPlannerWarningText ? sanitizePlanText(aiPlannerWarningText) : '请检查 API base、key、模型名，以及中转是否支持当前模型。'}`
          : 'AI 这次没有生成可执行方案。可以重试，或把目标、数据、模型写得更具体。'
  const guardParams = {
    ...toRecord(displayedVlaPlan?.params),
    ...(aiConfiguredParams || {}),
  }
  const guardDatasetSource = toRecord(guardParams.datasetSource)
  const guardModelSource = toRecord(guardParams.modelSource)
  const guardConfigName = firstString(
    guardParams.configName,
    guardParams.rlinfConfigName,
    guardParams.builtinTrainingProfile,
    toRecord(guardParams.rlinfConfig).configName,
  )
  const guardBenchmark = inferBenchmarkKey(
    guardParams.benchmark,
    guardParams.suite,
    guardParams.environmentKind,
    guardConfigName,
    cloudIntent,
    cloudBenchmark,
  )
  const guardDatasetText = firstString(
    guardDatasetSource.uri,
    guardDatasetSource.datasetId,
    guardDatasetSource.benchmark,
    guardParams.datasetPath,
    guardParams.dataPath,
    cloudSourceUri,
    cloudDatasetUri,
  ).toLowerCase()
  const benchmarkDatasetMismatch = Boolean(
    trainPlan
    && guardBenchmark === 'maniskill'
    && guardDatasetText.includes('libero')
  )
  const guardModelText = firstString(
    guardModelSource.uri,
    guardModelSource.checkpoint,
    guardModelSource.modelFamily,
    guardParams.modelFamily,
    guardParams.policyType,
    cloudModelUri,
    effectiveCloudPolicyType,
  )
  const unresolvedModelForLaunch = Boolean(
    trainPlan
    && cloudModelInitSource !== 'none'
    && !guardConfigName
    && isUnresolvedModelText(guardModelText)
  )
  const requiresRuntimeRebind = Boolean(!cloudBridgeChecking && (!cloudBridgeConnected || (cloudBridgeConnected && !cloudRuntimeReady)))
  const cloudSnapshotRestartCandidate = currentTrainMode === 'cloud'
    && Boolean(restartableCloudJobId)
    && !requiresRuntimeRebind
    && /旧实例|旧任务|重新启动任务|重新绑定云端实例|当前云端实例已能看到\s*GPU/.test(trainJobMessage)
  const shouldUseCloudPlanBlockers = !cloudSnapshotRestartCandidate
  const cloudStartBlockers = [
    ...(cloudBridgeChecking ? ['正在检查云端实例连接。'] : []),
    ...(cloudBridgeStatus?.enabled === false ? ['云训练后台还没有连接到团队的 EVO_Train 服务。这是部署配置，不需要用户填写。'] : []),
    ...(requiresRuntimeRebind ? ['云端实例未就绪，请重新连接。'] : []),
    ...(session.state !== 'idle' && session.state !== 'error' ? [`当前会话状态是 ${session.state}，需要先停止采集/回放/推理。`] : []),
    ...(shouldUseCloudPlanBlockers && cloudSourceKind === 'official_reference' && !officialDatasetSource ? ['当前 benchmark 还没有内置官方数据集映射，请切换到公开数据链接或平台数据。'] : []),
    ...(shouldUseCloudPlanBlockers && benchmarkDatasetMismatch ? ['当前任务是 ManiSkill，但数据仍是 LIBERO。请重新发送指令，或明确指定 ManiSkill 数据/环境。'] : []),
    ...(shouldUseCloudPlanBlockers && unresolvedModelForLaunch ? ['模型还没有解析成具体模型、checkpoint 或 RLinf config，不能启动。'] : []),
    ...(shouldUseCloudPlanBlockers && cloudSourceKind === 'platform_dataset' && !selectedCloudDataset ? ['请选择一个 Evo Studio 数据，或者切换到官方开源数据集、公开数据链接、我的云端数据。'] : []),
    ...(shouldUseCloudPlanBlockers && cloudSourceKind !== 'platform_dataset' && cloudSourceKind !== 'official_reference' && !trimmedCloudUri ? ['请填写数据来源地址。'] : []),
    ...(shouldUseCloudPlanBlockers && cloudModelInitSource === 'external' && !cloudModelUri.trim() ? ['请填写外部模型地址，或改为 AI 自动判断/不加载预训练权重。'] : []),
    ...(shouldUseCloudPlanBlockers && cloudDataAccessMode === 'saved_connection' && !cloudAuthRef.trim() ? ['请选择数据授权连接，或改为公开/无需授权。'] : []),
    ...(shouldUseCloudPlanBlockers && cloudModelAccessMode === 'saved_connection' && !cloudModelAuthRef.trim() ? ['请选择模型授权连接，或改为公开/无需授权。'] : []),
    ...(shouldUseCloudPlanBlockers && cloudExecutionMode === 'gpu' && isExistingSshRuntime && !cloudGpuReady ? ['当前 SSH 实例没有可见 GPU。请开卡/重新绑定有卡实例，或切到“总控自动/只准备”先无卡准备环境。'] : []),
    ...(shouldUseCloudPlanBlockers && !isExistingSshRuntime && selectedSkuSoldOut ? ['当前选择的机型暂无库存，请换一个机型或使用平台推荐。'] : []),
    ...(shouldUseCloudPlanBlockers && !isExistingSshRuntime && selectedSkuTooSmallForContainer ? [`当前机型是 ${selectedSkuGpuCount} 卡实例，但你选择了单实例 ${cloudGpuCount} 卡。请换多卡实例或降低 GPU 数。`] : []),
    ...(shouldUseCloudPlanBlockers && !isExistingSshRuntime && !selectedSkuTooSmallForContainer && resourceRequestTooLarge ? [`当前请求 ${requestedGpuTotal} 张卡，超过平台看到的空闲卡 ${selectedAvailableGpuCount}。请减少 GPU 数量/副本数或换机型。`] : []),
    ...(shouldUseCloudPlanBlockers && runtimeIsIncompatible ? [`当前机型/镜像暂不能启动：${runtimeBlockingMessages[0] || '请选择其他资源。'}`] : []),
  ]
  const localStartBlockers = [
    ...(session.state !== 'idle' && session.state !== 'error' ? [`当前会话状态是 ${session.state}，需要先停止采集/回放/推理。`] : []),
    ...(!trainDataset ? ['请选择一个本地可训练数据集。'] : []),
  ]
  const startBlockers = trainMode === 'cloud' ? cloudStartBlockers : localStartBlockers
  const savePrivateAuthConnection = async () => {
    const cleanId = authConnectionId.trim()
    const cleanProvider = authConnectionProvider.trim()
    const secrets: Record<string, string> = usesObjectStorageSecret
      ? {
          accessKeyId: authConnectionAccessKey.trim(),
          secretAccessKey: authConnectionSecretKey.trim(),
        }
      : { token: authConnectionToken.trim() }
    const hasSecret = Object.values(secrets).every(Boolean)
    if (!cleanId) {
      setAuthConnectionMessage('请先填写连接 ID。')
      return
    }
    if (!hasSecret) {
      setAuthConnectionMessage('请填写这个私有连接需要的 token 或 access key。')
      return
    }
    const connection = await saveAuthConnection({
      username: cloudUsername,
      id: cleanId,
      kind: authConnectionKind,
      provider: cleanProvider,
      label: authConnectionLabel.trim() || cleanId,
      visibility: 'user',
      secrets,
    })
    if (!connection) return
    setAuthConnectionMessage(`已保存：${connection.label || connection.id}`)
    if (authConnectionKind === 'data') {
      setCloudDataAccessMode('saved_connection')
      setCloudAuthRef(connection.id)
    } else {
      setCloudModelAccessMode('saved_connection')
      setCloudModelAuthRef(connection.id)
    }
    setAuthConnectionToken('')
    setAuthConnectionAccessKey('')
    setAuthConnectionSecretKey('')
  }

  const bindSshRuntime = async () => {
    const command = sshBindCommand.trim()
    const password = sshBindPassword
    const keyPath = sshBindKeyPath.trim()
    if (!command) {
      setSshBindMessage('请先粘贴 SSH 命令。')
      return
    }
    if (!password && !keyPath) {
      setSshBindMessage('请输入密码，或填写私钥路径。')
      return
    }
    setSshBindLoading(true)
    setSshBindMessage('正在检查云端 SSH 连接...')
    try {
      const result = await postJson('/api/train/cloud/dev/rebind-ssh', {
        sshCommand: command,
        password,
        keyPath,
        restartBridge: true,
      }) as Record<string, any>
      const bridge = toRecord(result.bridge)
      setSshBindPassword('')
      const runtimeReady = result.ok === true && result.runtimeReady !== false && bridge.listening === true
      const endpoint = String(result.endpoint || [result.user, result.host].filter(Boolean).join('@') + (result.port ? `:${result.port}` : '') || command)
      if (runtimeReady) {
        detachCurrentTrainJob('已连接新的云端实例，旧任务已移到历史。')
      }
      const validationError = String(result.validationError || '')
      const previousEndpoint = String(result.previousEndpoint || '')
      const rolledBack = result.rolledBack === true
      const clearedStaleBinding = result.clearedStaleBinding === true
      const rollbackText = rolledBack && previousEndpoint && previousEndpoint !== endpoint
        ? `已回滚到上一绑定：${previousEndpoint}。`
        : rolledBack
          ? '当前绑定未改变。'
          : ''
      setSshBindMessage(runtimeReady
        ? `已连接云端实例：${endpoint}`
        : clearedStaleBinding
          ? `没有连上：${endpoint}，已自动清除这个旧绑定。${validationError || '请粘贴当前实例最新 SSH 命令。'}`
          : `没有连上：${endpoint}，未保存为新的当前实例。${rollbackText}${validationError || '后端检查未通过，请确认实例已开机并允许 SSH 登录。'}`)
      await loadCloudBridgeStatus()
      await loadCloudResources(cloudProvider)
    } catch (error) {
      setSshBindMessage(error instanceof Error ? `连接失败：${error.message}` : '连接失败')
    } finally {
      setSshBindLoading(false)
    }
  }

  const clearSshRuntimeBinding = async () => {
    setSshBindLoading(true)
    setSshBindMessage('正在清除旧实例绑定...')
    try {
      const result = await postJson('/api/train/cloud/dev/unbind-ssh', {}) as Record<string, any>
      const previousEndpoint = String(result.previousEndpoint || '')
      setSshBindCommand('')
      setSshBindPassword('')
      setSshBindMessage(previousEndpoint
        ? `已清除旧绑定：${previousEndpoint}。请粘贴当前实例最新 SSH 命令。`
        : '已清除旧绑定。请粘贴当前实例最新 SSH 命令。')
      await loadCloudBridgeStatus()
      await loadCloudResources(cloudProvider)
    } catch (error) {
      setSshBindMessage(error instanceof Error ? `清除失败：${error.message}` : '清除失败')
    } finally {
      setSshBindLoading(false)
    }
  }

  const publicSourceConfirmation = (confirmedAt = '') => {
    const confirmed = Boolean(confirmedAt || (publicSourceConfirmationKey && cloudSourceConfirmedKey === publicSourceConfirmationKey))
    return confirmed
      ? {
          userConfirmed: true,
          confirmedAt: confirmedAt || cloudSourceConfirmedAt,
          confirmedRisks: sourcePreflightRisks.length > 0
            ? sourcePreflightRisks
            : ['public_source_download', 'cloud_cost', 'license_responsibility'],
        }
      : {}
  }

  const buildCloudDatasetSource = (confirmedAt = ''): CloudDatasetSource => {
    const aiDatasetSource = toRecord((aiConfiguredParams || {}).datasetSource)
    const aiDatasetUri = firstString(aiDatasetSource.uri, aiDatasetSource.url, aiDatasetSource.path)
    const aiDatasetId = firstString(aiDatasetSource.datasetId, aiDatasetSource.id)
    const aiDatasetFormat = firstString(aiDatasetSource.format)
    if (cloudSourceKind === 'official_reference') {
      return {
        sourceType: 'public_reference',
        datasetId: aiDatasetId || cloudBenchmark,
        uri: officialDatasetSource?.uri || aiDatasetUri || '',
        format: officialDatasetSource?.format || aiDatasetFormat || inferredDatasetFormat,
        benchmark: firstString(aiDatasetSource.benchmark, cloudBenchmark),
      }
    }
    if (cloudSourceKind === 'platform_dataset') {
      return {
        sourceType: cloudSourceKind,
        datasetId: selectedCloudDataset?.id || '',
        uri: cloudDatasetUri,
        format: inferredDatasetFormat,
      }
    }
    const explicitPublicUri = cloudSourceKind === 'public_reference' ? normalizePublicDatasetUri(trimmedCloudUri) : trimmedCloudUri
    return {
      sourceType: cloudSourceKind,
      datasetId: aiDatasetId,
      uri: explicitPublicUri || aiDatasetUri,
      authRef: cloudDataAccessMode === 'saved_connection' ? cloudAuthRef.trim() : '',
      format: aiDatasetFormat || inferredDatasetFormat,
      benchmark: firstString(aiDatasetSource.benchmark),
      ...(cloudSourceKind === 'public_reference' ? publicSourceConfirmation(confirmedAt) : {}),
    }
  }

  const buildCloudModelSource = (): CloudModelSource => {
    if (cloudModelInitSource === 'none') {
      return {
        sourceType: 'from_scratch',
        modelFamily: effectiveCloudPolicyType,
        format: inferredCheckpointFormat,
      }
    }
    if (trimmedCloudModelUri) {
      const connectedModelSourceType: CloudModelSource['sourceType'] = isObjectStorageUri(cloudModelUri)
        ? 'user_object_storage'
        : 'public_model_repo'
      return {
        sourceType: connectedModelSourceType,
        modelFamily: effectiveCloudPolicyType,
        uri: trimmedCloudModelUri,
        authRef: cloudModelAccessMode === 'saved_connection' ? cloudModelAuthRef.trim() : '',
        format: inferredCheckpointFormat,
      }
    }
    const aiModelSource = toRecord(aiConfiguredParams?.modelSource)
    const aiModelSourceType = firstString(aiModelSource.sourceType, aiModelSource.type)
    const aiModelUri = firstString(aiModelSource.uri, aiModelSource.modelUri, aiModelSource.repoId, aiModelSource.repository)
    const aiModelCheckpoint = firstString(aiModelSource.checkpoint, aiModelSource.checkpointName)
    const aiModelAuthRef = firstString(aiModelSource.authRef, aiModelSource.credentialRef)
    const aiModelFormat = firstString(aiModelSource.format, aiConfiguredParams?.checkpointFormat) || inferredCheckpointFormat
    if (cloudModelInitSource === 'auto' && (aiModelUri || aiModelCheckpoint || aiModelSourceType)) {
      return {
        sourceType: (aiModelSourceType || 'ai_resolved') as CloudModelSource['sourceType'],
        modelFamily: effectiveCloudPolicyType,
        uri: aiModelUri,
        checkpoint: aiModelCheckpoint,
        authRef: aiModelAuthRef,
        format: aiModelFormat,
      }
    }
    return {
      sourceType: 'builtin_policy',
      modelFamily: effectiveCloudPolicyType,
      format: inferredCheckpointFormat,
    }
  }

  const environmentParams = cloudEnvironmentMode === 'benchmark'
    ? {
        benchmark: cloudBenchmark,
        suite: cloudBenchmark,
        environmentKind: cloudBenchmark === 'custom' ? 'custom_benchmark' : 'benchmark',
        environmentHint: cloudEnvironmentHint.trim(),
      }
    : { environmentKind: 'none' }
  const isRlinfExecution = cloudWorkflow === 'rlinf_vla'
  const effectiveRlinfAlgorithm = rlinfAlgorithm === 'auto'
    ? firstString(aiConfiguredParams?.algorithm, aiConfiguredParams?.algorithmName, aiConfiguredParams?.lossType)
    : rlinfAlgorithm
  const rlinfActorWorker = effectiveRlinfAlgorithm === 'sac'
    ? 'EmbodiedSACFSDPPolicy'
    : 'EmbodiedFSDPActor'
  const rlinfLauncherKind = effectiveRlinfAlgorithm && effectiveRlinfAlgorithm in RLINF_BACKEND_INTERFACE.algorithmToLauncherKind
    ? RLINF_BACKEND_INTERFACE.algorithmToLauncherKind[effectiveRlinfAlgorithm as keyof typeof RLINF_BACKEND_INTERFACE.algorithmToLauncherKind]
    : 'python_module'
  const buildRlinfRuntimeParams = () => {
    if (!isRlinfExecution) return {}
    return {
      backendKind: 'rlinf',
      recipeId: 'rlinf_vla_default',
      recipe: RLINF_RECIPE_CATALOG.rlinf_vla_default,
      backendInterface: RLINF_BACKEND_INTERFACE,
      launchMode: 'project_backend',
      launcherKind: rlinfLauncherKind,
      algorithm: effectiveRlinfAlgorithm || undefined,
      groupSize: effectiveRlinfAlgorithm === 'grpo' ? rlinfGroupSize : undefined,
      placementStrategy: rlinfPlacementStrategy,
      rolloutBackend: rlinfRolloutBackend,
      runnerKind: 'embodied_runner',
      actorWorker: rlinfActorWorker,
      rolloutWorker: 'MultiStepRolloutWorker',
      envWorker: 'EnvWorker',
      scheduler: 'Cluster',
      placementClass: 'HybridComponentPlacement',
      rlinfContract: {
        recipeId: 'rlinf_vla_default',
        framework: 'rlinf',
        runner: 'EmbodiedRunner',
        scheduler: 'Cluster',
        placement: {
          className: 'HybridComponentPlacement',
          strategy: rlinfPlacementStrategy,
          components: {
            actor: 'actor_group',
            rollout: 'rollout_group',
            env: 'env_group',
          },
        },
        workers: {
          actor: rlinfActorWorker,
          rollout: 'MultiStepRolloutWorker',
          env: 'EnvWorker',
        },
        algorithm: {
          name: effectiveRlinfAlgorithm || 'auto',
          groupSize: effectiveRlinfAlgorithm === 'grpo' ? rlinfGroupSize : undefined,
        },
        rollout: {
          backend: rlinfRolloutBackend,
        },
        registryInjection: RLINF_BACKEND_INTERFACE.registryInjection,
      },
    }
  }
  const buildManualTrainingOverrides = () => {
    const overrides: Record<string, unknown> = {
      steps: trainSteps,
      maxSteps: trainSteps,
      device: trainDevice,
    }
    const learningRate = parseManualParamValue(cloudLearningRate)
    if (learningRate !== undefined) {
      overrides.learningRate = learningRate
      overrides.lr = learningRate
    }
    const batchSize = parseManualParamValue(cloudBatchSize)
    if (batchSize !== undefined) overrides.batchSize = batchSize
    const epochs = parseManualParamValue(cloudEpochs)
    if (epochs !== undefined) overrides.epochs = epochs
    const warmupSteps = parseManualParamValue(cloudWarmupSteps)
    if (warmupSteps !== undefined) overrides.warmupSteps = warmupSteps
    const gradientAccumulationSteps = parseManualParamValue(cloudGradientAccumulationSteps)
    if (gradientAccumulationSteps !== undefined) {
      overrides.gradientAccumulationSteps = gradientAccumulationSteps
    }
    const loraRank = parseManualParamValue(cloudLoraRank)
    if (loraRank !== undefined) overrides.loraRank = loraRank
    if (cloudExecutionNotes.trim()) {
      overrides.customExecutionInstructions = cloudExecutionNotes.trim()
      overrides.userExecutionRequest = cloudExecutionNotes.trim()
    }
    return overrides
  }
  const manualTrainingOverrides = buildManualTrainingOverrides()

  const skuForGpuCount = (count: number) => {
    const familyKey = selectedSkuFamilyKey || skuFamilyKey(selectedCloudSku) || skuFamilyKey(selectedCloudSkuRaw)
    if (!familyKey) return null
    return regionFilteredCloudSkus.find(sku => (
      skuFamilyKey(sku) === familyKey
      && Number(sku.gpuCount || 1) === count
      && sku.stockStatus !== 'sold_out'
      && skuAvailableInRegion(sku, cloudRegionSign)
    )) || null
  }

  useEffect(() => {
    const readySku = regionFilteredCloudSkus.find((sku) => {
      if (sku.readyToStart === false) return false
      const skuGpuCount = Number(sku.gpuCount || 1)
      const stockCount = skuRegionCapacity(sku, cloudRegionSign).stockCount
      return skuGpuCount >= cloudGpuCount && (!stockCount || stockCount >= cloudReplicaCount)
    }) || regionFilteredCloudSkus.find(sku => sku.readyToStart !== false)
    const readyImage = (cloudResourceCatalog?.images || []).find(image => image.readyToStart !== false)
    setCloudSkuId(current => {
      if (current && regionFilteredCloudSkus.some(sku => sku.skuId === current)) {
        const currentSku = regionFilteredCloudSkus.find(sku => sku.skuId === current)
        const familyKey = skuFamilyKey(currentSku)
        const exactSku = regionFilteredCloudSkus.find(sku => (
          skuFamilyKey(sku) === familyKey
          && Number(sku.gpuCount || 1) === cloudGpuCount
          && sku.stockStatus !== 'sold_out'
          && sku.readyToStart !== false
          && skuAvailableInRegion(sku, cloudRegionSign)
        ))
        return String(exactSku?.skuId || current)
      }
      return readySku?.skuId || ''
    })
    setCloudImageId(current => {
      if (current && (cloudResourceCatalog?.images || []).some(image => image.imageId === current)) return current
      return readyImage?.imageId || ''
    })
  }, [cloudResourceCatalog, cloudRegionSign, cloudGpuCount, cloudReplicaCount])

  useEffect(() => {
    if (cloudSourceConfirmedKey && cloudSourceConfirmedKey !== publicSourceConfirmationKey) {
      setCloudSourceConfirmedKey('')
      setCloudSourceConfirmedAt('')
    }
  }, [cloudSourceConfirmedKey, publicSourceConfirmationKey])

  useEffect(() => {
    if (cloudSourceKind !== 'public_reference' || !normalizedPublicDatasetUri) return
    const timer = window.setTimeout(() => {
      void loadSourcePreflight({
        username: cloudUsername.trim(),
        provider: cloudProvider,
        role: 'dataset',
        source: {
          sourceType: 'public_reference',
          uri: normalizedPublicDatasetUri,
          format: inferredDatasetFormat,
        },
      })
    }, 300)
    return () => window.clearTimeout(timer)
  }, [
    cloudSourceKind,
    normalizedPublicDatasetUri,
    inferredDatasetFormat,
    cloudUsername,
    cloudProvider,
    loadSourcePreflight,
  ])

  useEffect(() => {
    if (cloudBridgeStatus?.enabled === false) return
    const timer = window.setTimeout(() => {
      const datasetSource = buildCloudDatasetSource()
      const modelSource = buildCloudModelSource()
      void loadRuntimeMatch({
        username: cloudUsername.trim(),
        provider: cloudProvider,
        sku_id: effectiveCloudSkuId,
        image_id: effectiveCloudImageId,
        force_refresh: true,
        params: {
          datasetSource,
          modelSource,
          datasetPath: datasetSource.uri,
          checkpointPath: modelSource.uri || modelSource.checkpoint,
          datasetFormat: inferredDatasetFormat,
          checkpointFormat: inferredCheckpointFormat,
          automationPolicy: cloudAutomationPolicy(cloudAutomationMode),
          policyType: effectiveCloudPolicyType,
          modelFamily: effectiveCloudPolicyType,
          ...buildRlinfRuntimeParams(),
          allowUnverifiedRuntime: false,
          gpuCount: cloudGpuCount,
          replicas: cloudReplicaCount,
          requestedGpuTotal,
          autodlRegionSign: cloudRegionSign || undefined,
          autodlDataCenters: cloudRegionSign || undefined,
          ...environmentParams,
          steps: trainSteps,
          device: trainDevice,
          ...manualTrainingOverrides,
        },
      })
    }, 250)
    return () => window.clearTimeout(timer)
  }, [
    cloudBridgeStatus?.enabled,
    cloudUsername,
    cloudProvider,
    effectiveCloudSkuId,
    effectiveCloudImageId,
    cloudWorkflow,
    policyType,
    effectiveCloudPolicyType,
    trainSteps,
    trainDevice,
    cloudGpuCount,
    cloudReplicaCount,
    cloudRegionSign,
    requestedGpuTotal,
    cloudSourceKind,
    cloudDatasetId,
    cloudSourceUri,
    cloudDataAccessMode,
    cloudAuthRef,
    cloudFormat,
    cloudEnvironmentMode,
    cloudBenchmark,
    cloudEnvironmentHint,
    cloudModelUri,
    cloudModelInitSource,
    cloudModelAccessMode,
    cloudModelAuthRef,
    cloudCheckpointFormat,
    cloudLearningRate,
    cloudBatchSize,
    cloudEpochs,
    cloudWarmupSteps,
    cloudGradientAccumulationSteps,
    cloudLoraRank,
    cloudAutomationMode,
    rlinfAlgorithm,
    rlinfPlacementStrategy,
    rlinfRolloutBackend,
    rlinfGroupSize,
    cloudBridgeStatus?.enabled,
    loadRuntimeMatch,
  ])

  const applyVlaPlanToForm = (vlaPlan: any) => {
    const params = toRecord(vlaPlan?.params)
    const datasetSource = toRecord(params.datasetSource)
    const modelSource = toRecord(params.modelSource)
    setAiConfiguredParams(params)

    const nextProvider = firstString(vlaPlan?.provider, params.provider, toRecord(params.runtime).provider)
    if (nextProvider === 'autodl' || nextProvider === 'aliyun') setCloudProvider(nextProvider)

    const nextWorkflow = firstString(vlaPlan?.workflow, params.workflow)
    if (nextWorkflow) setCloudWorkflow(nextWorkflow)
    const nextAlgorithm = firstString(params.algorithm, params.algorithmName, params.lossType)
    if (nextAlgorithm) setRlinfAlgorithm(nextAlgorithm)
    const nextPlacement = firstString(params.placementStrategy, toRecord(params.rlinfContract).placement?.strategy)
    if (nextPlacement) setRlinfPlacementStrategy(nextPlacement)
    const nextRolloutBackend = firstString(params.rolloutBackend, toRecord(params.rlinfContract).rollout?.backend)
    if (nextRolloutBackend) setRlinfRolloutBackend(nextRolloutBackend)
    const nextGroupSize = Number(params.groupSize ?? toRecord(params.algorithm).groupSize)
    if (Number.isFinite(nextGroupSize) && nextGroupSize > 0) setRlinfGroupSize(nextGroupSize)

    const nextModelFamily = canonicalPolicyType(firstString(
      params.modelFamily,
      params.policyType,
      modelSource.modelFamily,
      modelSource.policyType,
    ))
    if (nextModelFamily) setPolicyType(nextModelFamily)

    const nextSteps = Number(params.steps ?? params.maxSteps ?? params.trainSteps)
    if (Number.isFinite(nextSteps) && nextSteps > 0) setTrainSteps(nextSteps)
    const nextDevice = firstString(params.device)
    if (nextDevice) setTrainDevice(nextDevice)
    const nextLearningRate = formatParamValue(firstValue(params, ['learningRate', 'lr', 'optimizer.learningRate', 'optimizer.lr']))
    if (nextLearningRate) setCloudLearningRate(nextLearningRate)
    const nextBatchSize = formatParamValue(firstValue(params, ['batchSize', 'globalBatchSize', 'trainBatchSize', 'perDeviceBatchSize', 'training.batchSize']))
    if (nextBatchSize) setCloudBatchSize(nextBatchSize)
    const nextEpochs = formatParamValue(firstValue(params, ['epochs', 'numEpochs', 'training.epochs']))
    if (nextEpochs) setCloudEpochs(nextEpochs)
    const nextWarmupSteps = formatParamValue(firstValue(params, ['warmupSteps', 'scheduler.warmupSteps']))
    if (nextWarmupSteps) setCloudWarmupSteps(nextWarmupSteps)
    const nextGradientAccumulation = formatParamValue(firstValue(params, ['gradientAccumulationSteps', 'gradientAccumulation', 'training.gradientAccumulationSteps']))
    if (nextGradientAccumulation) setCloudGradientAccumulationSteps(nextGradientAccumulation)
    const nextLoraRank = formatParamValue(firstValue(params, ['loraRank', 'lora.rank', 'adapter.loraRank']))
    if (nextLoraRank) setCloudLoraRank(nextLoraRank)

    const datasetFormat = firstString(params.datasetFormat, datasetSource.format)
    if (datasetFormat) setCloudFormat(datasetFormat)
    const checkpointFormat = firstString(params.checkpointFormat, modelSource.format)
    if (checkpointFormat) setCloudCheckpointFormat(checkpointFormat)

    if (Object.keys(datasetSource).length > 0) {
      const sourceKind = toCloudSourceKind(datasetSource.sourceType)
      setCloudSourceKind(sourceKind)
      const datasetId = firstString(datasetSource.datasetId, datasetSource.id)
      const sourceUri = firstString(datasetSource.uri, datasetSource.url, datasetSource.path)
      if (sourceKind === 'platform_dataset') {
        if (datasetId) setCloudDatasetId(datasetId)
        if (!datasetId && sourceUri) {
          setCloudSourceKind('user_object_storage')
          setCloudSourceUri(sourceUri)
        }
      } else {
        setCloudSourceUri(sourceUri)
      }
      const authRef = firstString(datasetSource.authRef, datasetSource.credentialRef)
      if (authRef) {
        setCloudDataAccessMode('saved_connection')
        setCloudAuthRef(authRef)
      } else {
        setCloudDataAccessMode('public')
      }
    } else {
      const datasetPath = firstString(params.datasetPath, params.dataPath, params.datasetUri)
      if (datasetPath) {
        setCloudSourceKind(datasetPath.startsWith('hf://') || datasetPath.includes('huggingface.co') ? 'public_reference' : 'user_object_storage')
        setCloudSourceUri(datasetPath)
      }
    }

    if (Object.keys(modelSource).length > 0) {
      const modelSourceType = firstString(modelSource.sourceType).toLowerCase()
      const modelUri = firstString(modelSource.uri, modelSource.url, modelSource.path)
      if (modelSourceType === 'from_scratch') {
        setCloudModelInitSource('none')
        setCloudModelUri('')
      }
      if (modelUri) {
        setCloudModelInitSource('external')
        setCloudModelUri(modelUri)
      } else if (modelSourceType === 'builtin_policy' || modelSourceType === 'ai_resolved') {
        setCloudModelInitSource('auto')
      }
      const modelAuthRef = firstString(modelSource.authRef, modelSource.credentialRef)
      if (modelAuthRef) {
        setCloudModelAccessMode('saved_connection')
        setCloudModelAuthRef(modelAuthRef)
      } else {
        setCloudModelAccessMode('public')
      }
    } else {
      const checkpointPath = firstString(params.checkpointPath, params.modelPath, params.modelCheckpointPath)
      if (checkpointPath) setCloudModelUri(checkpointPath)
    }

    const rlinfConfigName = firstString(params.configName, params.rlinfConfigName, params.builtinTrainingProfile)
    const inferredPlanBenchmark = inferBenchmarkKey(
      params.benchmark,
      params.suite,
      params.environmentKind,
      rlinfConfigName,
      vlaPlan?.aiSummary,
    )
    const environmentKind = firstString(params.environmentKind, params.benchmark, params.suite, inferredPlanBenchmark).toLowerCase()
    if (environmentKind && environmentKind !== 'none') {
      setCloudEnvironmentMode('benchmark')
      const benchmark = firstString(inferredPlanBenchmark, params.benchmark, params.suite, params.environmentKind)
      if (benchmark) setCloudBenchmark(benchmark.toLowerCase())
    }
    const environmentHint = firstString(params.environmentHint, params.liberoTaskOrSuite, params.taskName)
    if (environmentHint) setCloudEnvironmentHint(environmentHint)
    const plannedRegionSign = firstString(
      params.autodlRegionSign,
      params.regionSign,
      toRecord(params.runtime).regionSign,
      toRecord(params.runtime).autodlRegionSign,
    )
    if (plannedRegionSign) setCloudRegionSign(plannedRegionSign)
    const plannedSkuId = firstString(params.skuId, params.sku_id)
    const plannedImageId = firstString(params.imageId, params.image_id)
    if (plannedSkuId) setCloudSkuId(plannedSkuId)
    if (plannedImageId) setCloudImageId(plannedImageId)

  }

  useEffect(() => {
    if (!trainPlan) return
    const vlaPlan = extractVlaPlanPayload(trainPlan)
    if (Object.keys(vlaPlan).length === 0) return
    const planKey = JSON.stringify({
      workflow: vlaPlan.workflow || '',
      runtimeMode: firstString((trainPlan as any).runtimeMode, vlaPlan.runtimeMode),
      params: vlaPlan.params || {},
    })
    if (!planKey || planKey === lastAppliedPlanKey) return
    applyVlaPlanToForm(vlaPlan)
    setLastAppliedPlanKey(planKey)
  }, [trainPlan, lastAppliedPlanKey])

  const confirmPublicSourceIfNeeded = () => {
    const aiDatasetSource = toRecord((aiConfiguredParams || {}).datasetSource)
    const publicDatasetUri = normalizedPublicDatasetUri || firstString(aiDatasetSource.uri, aiDatasetSource.url, aiDatasetSource.path)
    if (cloudSourceKind !== 'public_reference' || !publicDatasetUri) return ''
    if (cloudSourceConfirmedKey === publicSourceConfirmationKey && cloudSourceConfirmedAt) {
      return cloudSourceConfirmedAt
    }
    const confirmedAt = new Date().toISOString()
    setCloudSourceConfirmedKey(publicSourceConfirmationKey)
    setCloudSourceConfirmedAt(confirmedAt)
    return confirmedAt
  }

  const buildTrainingContract = (
    datasetSource: CloudDatasetSource,
    modelSource: CloudModelSource,
  ) => {
    const aiParams = aiConfiguredParams || {}
    const automationPolicy = cloudAutomationPolicy(cloudAutomationMode)
    const algorithmName = firstString(effectiveRlinfAlgorithm, aiParams.algorithm, aiParams.algorithmName, aiParams.lossType)
    const envName = cloudEnvironmentMode === 'benchmark'
      ? cloudBenchmark
      : firstString(datasetSource.datasetId, selectedCloudDataset?.id, cloudBenchmark)
    const rlinfRuntimeParams = buildRlinfRuntimeParams() as Record<string, any>
    return {
      interfaceKind: cloudWorkflow === 'rlinf_vla' ? 'rlinf_runner' : 'project_launcher',
      framework: cloudWorkflow === 'rlinf_vla' ? 'rlinf' : 'project_backend',
      sources: {
        dataset: datasetSource,
        model: modelSource,
      },
      runner: {
        maxSteps: trainSteps,
        device: trainDevice,
        smokeFirst: true,
      },
      actor: {
        model: {
          family: effectiveCloudPolicyType,
          checkpointFormat: inferredCheckpointFormat,
          sourceType: modelSource.sourceType,
        },
      },
      rollout: {
        replicas: cloudReplicaCount,
        placementStrategy: cloudWorkflow === 'rlinf_vla' ? rlinfPlacementStrategy : 'single_node',
        backend: cloudWorkflow === 'rlinf_vla' ? rlinfRolloutBackend : undefined,
      },
      env: {
        kind: cloudEnvironmentMode === 'benchmark' ? 'benchmark' : 'dataset_only',
        name: envName,
        datasetFormat: inferredDatasetFormat,
        hint: cloudEnvironmentHint.trim(),
      },
      algorithm: {
        name: algorithmName || (cloudWorkflow === 'rlinf_vla' ? 'auto' : ''),
        groupSize: cloudWorkflow === 'rlinf_vla' && effectiveRlinfAlgorithm === 'grpo' ? rlinfGroupSize : undefined,
      },
      runtime: {
        provider: cloudProvider,
        runtimeMode: isExistingSshRuntime ? 'ssh_existing_instance' : '',
        existingInstance: isExistingSshRuntime,
        automationPolicy,
        regionSign: cloudRegionSign || '',
        regionLabel: cloudRegionSign ? autodlRegionLabel(cloudRegionSign) : '平台自动择区',
        gpuCount: cloudGpuCount,
        replicas: cloudReplicaCount,
        requestedGpuTotal,
        skuId: effectiveCloudSkuId,
        imageId: effectiveCloudImageId,
      },
      execution: cloudExecutionNotes.trim()
        ? {
            userInstructions: cloudExecutionNotes.trim(),
          }
        : undefined,
      rlinf: cloudWorkflow === 'rlinf_vla' ? rlinfRuntimeParams.rlinfContract : undefined,
    }
  }

  const startCloudTraining = () => {
    try {
      useTrainingStore.setState({ trainJobMessage: '正在提交云训练...' })
      const confirmedAt = confirmPublicSourceIfNeeded()
      const datasetSource = buildCloudDatasetSource(confirmedAt)
      const modelSource = buildCloudModelSource()
      if (cloudSourceKind === 'public_reference' && !datasetSource.uri) {
        useTrainingStore.setState({ trainJobMessage: '云训练启动失败：数据集链接为空，请重新生成方案或填写数据集链接。' })
        return
      }
      const trainingContract = buildTrainingContract(datasetSource, modelSource)
      const datasetLabel = officialDatasetSource?.label || selectedCloudDataset?.label || datasetSource.uri || cloudBenchmark || 'cloud-source'
      const taskRunSuffix = Date.now().toString(36)
      const prepareOnly = cloudExecutionMode === 'prepare' || autoPrepareOnNoGpu
      const executionPhase = prepareOnly ? 'prepare_only' : cloudExecutionMode === 'gpu' ? 'run' : 'auto'
      if (typeof localStorage !== 'undefined' && cloudUsername.trim()) {
        localStorage.setItem('roboclaw.dataset.username', cloudUsername.trim())
      }
      if (typeof window !== 'undefined' && !isExistingSshRuntime) {
        const costLabel = prepareOnly
          ? '无卡准备：不冻结 GPU 首小时费用'
          : Number.isFinite(selectedHourlyCostCents) && selectedHourlyCostCents > 0
          ? `约 ¥${(selectedHourlyCostCents / 100).toFixed(2)}/小时，最终以云平台返回为准`
          : '费用以云平台返回为准'
        const startDetail = [
          '即将把目标交给 Evo Studio 实验总控；总控会安排数据准备、模型配置、云端运行和结果产物。',
          `数据：${datasetLabel}`,
          `模型：${modelSource.uri || modelSource.checkpoint || modelSource.modelFamily || effectiveCloudPolicyType}`,
          `模式：${prepareOnly ? '无卡准备，只做环境/代码/数据/模型缓存' : cloudExecutionMode === 'gpu' ? 'GPU 运行，进入训练或评测' : '总控自动判断'}`,
          isExistingSshRuntime
            ? '资源：后端已配置的 SSH GPU 实例'
            : `资源：${selectedCloudSku ? String(selectedCloudSku.displayName || selectedCloudSku.skuId) : '平台推荐'}，单实例 ${cloudGpuCount} 张 GPU × ${cloudReplicaCount} 个副本`,
          !isExistingSshRuntime && selectedCloudImage ? `镜像：${String(selectedCloudImage.displayName || selectedCloudImage.imageId)}` : '',
          `费用：${costLabel}`,
          '确认启动？',
        ].filter(Boolean).join('\n')
        if (!window.confirm(startDetail)) {
          useTrainingStore.setState({ trainJobMessage: '已取消启动。' })
          return
        }
      }
      void doCloudTrainStart({
        username: cloudUsername.trim(),
        provider: cloudProvider,
        sku_id: effectiveCloudSkuId,
        image_id: effectiveCloudImageId,
        hourly_cost_cents: prepareOnly ? 0 : Number.isFinite(selectedHourlyCostCents) ? selectedHourlyCostCents : 0,
        dataset_name: cloudSourceKind === 'platform_dataset'
          ? (selectedCloudDataset?.runtime?.name || selectedCloudDataset?.id || '')
          : '',
        workflow: cloudWorkflow.trim() || 'rlinf_vla',
        policy_type: effectiveCloudPolicyType,
        steps: trainSteps,
        device: trainDevice,
        waitForSubmit: true,
        wait_for_submit: true,
        task_name: `cloud-${effectiveCloudPolicyType}-${String(datasetLabel).replace(/[^a-zA-Z0-9_.-]+/g, '-').slice(0, 28)}-${taskRunSuffix}`,
        params: {
          ...(aiConfiguredParams || {}),
          trainingContract,
          datasetSource,
          modelSource,
          datasetPath: datasetSource.uri,
          checkpointPath: modelSource.uri || modelSource.checkpoint,
          datasetFormat: inferredDatasetFormat,
          checkpointFormat: inferredCheckpointFormat,
          automationPolicy: cloudAutomationPolicy(cloudAutomationMode),
          executionPhase,
          runPhase: executionPhase,
          prepareOnly,
          gpuRequired: cloudExecutionMode === 'gpu' ? true : prepareOnly ? false : undefined,
          policyType: effectiveCloudPolicyType,
          modelFamily: effectiveCloudPolicyType,
          ...manualTrainingOverrides,
          ...buildRlinfRuntimeParams(),
          resourceIntent: 'auto',
          allowUnverifiedRuntime: false,
          runtimeMode: isExistingSshRuntime ? 'ssh_existing_instance' : undefined,
          useExistingInstance: isExistingSshRuntime || undefined,
          gpuCount: cloudGpuCount,
          replicas: cloudReplicaCount,
          requestedGpuTotal,
          autodlRegionSign: cloudRegionSign || undefined,
          autodlDataCenters: cloudRegionSign || undefined,
          ...environmentParams,
          steps: trainSteps,
          device: trainDevice,
          ...manualTrainingOverrides,
        },
      }).then(() => {
        clearCloudTrainPlan()
        setAiConfiguredParams(null)
      }).catch((error) => {
        useTrainingStore.setState({
          trainJobMessage: error instanceof Error ? `云训练启动失败：${error.message}` : '云训练启动失败',
        })
      })
    } catch (error) {
      useTrainingStore.setState({
        trainJobMessage: error instanceof Error ? `云训练启动失败：${error.message}` : '云训练启动失败',
      })
    }
  }

  const currentCloudJobStatus = parseTrainStatusFromMessage(trainJobMessage)
  const activeCloudJob = currentTrainMode === 'cloud' && Boolean(currentTrainJobId) && isActiveCloudStatus(currentCloudJobStatus)
  const cloudRestartReady = cloudSnapshotRestartCandidate && !activeCloudJob
  const cloudIntentBusy = activeCloudJob || trainingPlanLoading || trainingLoading
  const shouldQueueCloudIntent = !trainPlan && Boolean(cloudIntent.trim()) && cloudIntentBusy
  const cloudPlanActionDisabled = !cloudIntent.trim()
  const cloudStartDisabled = cloudBridgeChecking
    ? true
    : activeCloudJob
    ? !shouldQueueCloudIntent
    : (trainPlan
        ? trainingLoading
        : (cloudPlanActionDisabled || trainingPlanLoading))
  const cloudStartLabel = cloudBridgeChecking
    ? '检查连接'
    : requiresRuntimeRebind
    ? '连接实例'
        : (trainPlan
            ? (aiPlannerUnavailable
                ? '重新理解目标'
                : (trainingLoading
                    ? '启动中...'
                  : cloudRestartReady
                    ? '继续运行'
                : cloudExecutionMode === 'prepare' || autoPrepareOnNoGpu
                  ? '开始无卡准备'
                  : cloudExecutionMode === 'gpu'
                    ? '开始 GPU 任务'
                  : '开始云端任务'))
        : (cloudRestartReady
            ? (restartableCloudJobId ? '继续运行' : '重新生成方案')
            : (shouldQueueCloudIntent ? '加入队列' : trainingPlanLoading ? '处理中...' : '发送目标')))
  const showCloudProgress = trainMode === 'cloud' && (
    activeCloudJob ||
    Boolean(trainJobMessage) ||
    trainJobHistory.length > 0
  )
  const handleCloudPrimaryAction = () => {
    const now = Date.now()
    if (now - cloudStartActivationRef.current < 700) return
    cloudStartActivationRef.current = now
    if (shouldQueueCloudIntent) {
      enqueueCloudTask(cloudIntent)
      return
    }
    if (activeCloudJob) return
    if (requiresRuntimeRebind) {
      useTrainingStore.setState({ trainJobMessage: '云端实例未就绪，请先连接实例。' })
      setShowSshRuntimeBind(true)
      window.setTimeout(() => {
        sshBindSectionRef.current?.scrollIntoView({ block: 'start', behavior: 'smooth' })
      }, 80)
      return
    }
    if (cloudRestartReady && restartableCloudJobId) {
      useTrainingStore.setState({ trainJobMessage: '正在复用上次任务参数继续运行...' })
      void repairCloudTraining({
        username: cloudUsername.trim(),
        job_id: restartableCloudJobId,
        user_guidance: cloudIntent.trim(),
      })
      return
    }
    if (trainPlan && aiPlannerUnavailable) {
      useTrainingStore.setState({ trainJobMessage: '正在重新让 AI 理解目标...' })
      void planCloudTraining()
      return
    }
    if (trainPlan && startBlockers.length > 0) {
      useTrainingStore.setState({ trainJobMessage: `还不能启动：${startBlockers[0]}` })
      return
    }
    if (trainPlan && !canStartTraining) {
      useTrainingStore.setState({ trainJobMessage: '还不能启动：当前会话还没有进入可提交状态。' })
      return
    }
    if (trainPlan && !aiPlannerUnavailable) {
      startCloudTraining()
    } else {
      void planCloudTraining()
    }
  }
  const renderCloudStartButton = (className = '') => {
    const iconOnly = !trainPlan && !cloudRestartReady
    const title = trainPlan
      ? (requiresRuntimeRebind
          ? '当前云端实例不可达，请重新绑定后再启动。'
          : (aiPlannerUnavailable
              ? 'AI 没有生成可用方案，请重新规划。'
                  : (activeCloudJob
                      ? '当前已有云任务在运行。'
                      : cloudRestartReady
                    ? '当前实例已恢复；点击会复用上次任务参数继续运行。'
                    : (startBlockers[0] || '开始云端任务'))))
      : (!cloudIntent.trim()
          ? '先写实验目标。'
          : cloudRestartReady
            ? '当前实例已恢复；重新发送目标后启动新任务。'
            : shouldQueueCloudIntent
              ? '当前任务未结束，先加入待处理队列。'
              : '发送给实验总控')
    return (
      <button
        type="button"
        disabled={cloudStartDisabled}
        data-cloud-start-button="true"
        aria-label={iconOnly ? cloudStartLabel : undefined}
        title={title}
        onClick={handleCloudPrimaryAction}
        className={`${iconOnly ? 'h-11 w-11 shrink-0 rounded-full grid place-items-center' : 'px-4 py-2.5 rounded-lg'}
          text-sm font-semibold text-white bg-ac hover:bg-ac2 shadow-glow-ac transition-all active:scale-[0.97]
          disabled:opacity-35 disabled:cursor-not-allowed disabled:shadow-none ${className}`}
      >
        {iconOnly ? <SendArrowIcon /> : cloudStartLabel}
      </button>
    )
  }
  const renderCloudSafetyControls = (buttonClassName = '') => (
    <div className="flex flex-wrap items-center gap-2">
      {!activeCloudJob && renderCloudStartButton(buttonClassName)}
      {cloudCheckLoading && (
        <button
          type="button"
          onClick={cancelCloudChecks}
          className="px-3 py-2.5 rounded-lg border border-bd bg-sf text-sm font-semibold text-tx hover:border-rd/60 hover:text-rd
            transition-all active:scale-[0.97]"
        >
          取消检查
        </button>
      )}
      {trainingLoading && (
        <button
          type="button"
          onClick={cancelCloudStartRequest}
          className="px-3 py-2.5 rounded-lg border border-yl/50 bg-yl/10 text-sm font-semibold text-tx hover:border-rd/60 hover:text-rd
            transition-all active:scale-[0.97]"
        >
          取消启动等待
        </button>
      )}
      {activeCloudJob && (
        <button
          type="button"
          disabled={trainingStopLoading}
          onClick={() => { void doTrainStop() }}
          className="px-3 py-2.5 rounded-lg border border-rd/60 bg-rd/10 text-sm font-semibold text-rd hover:bg-rd/15
            transition-all active:scale-[0.97] disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {trainingStopLoading ? '停止中...' : '停止云训练'}
        </button>
      )}
    </div>
  )

  const planCloudTraining = async (intentOverride?: string) => {
    const intentText = (intentOverride ?? cloudIntent).trim()
    if (!intentText) return
    const datasetSource = buildCloudDatasetSource()
    const modelSource = buildCloudModelSource()
    const trainingContract = buildTrainingContract(datasetSource, modelSource)
    const data = await doCloudTrainPlan({
      username: cloudUsername.trim(),
      provider: cloudProvider,
      message: [
        intentText,
        cloudExecutionNotes.trim() ? `补充执行说明：${cloudExecutionNotes.trim()}` : '',
      ].filter(Boolean).join('\n\n'),
      workflow: cloudWorkflow.trim(),
      sku_id: effectiveCloudSkuId,
      image_id: effectiveCloudImageId,
      params: {
        trainingContract,
        datasetSource,
        modelSource,
        datasetPath: datasetSource.uri,
        checkpointPath: modelSource.uri || modelSource.checkpoint,
        datasetFormat: inferredDatasetFormat,
        checkpointFormat: inferredCheckpointFormat,
        automationPolicy: cloudAutomationPolicy(cloudAutomationMode),
        executionPhase: cloudExecutionMode === 'prepare' || autoPrepareOnNoGpu ? 'prepare_only' : cloudExecutionMode === 'gpu' ? 'run' : 'auto',
        runPhase: cloudExecutionMode === 'prepare' || autoPrepareOnNoGpu ? 'prepare_only' : cloudExecutionMode === 'gpu' ? 'run' : 'auto',
        prepareOnly: cloudExecutionMode === 'prepare' || autoPrepareOnNoGpu,
        gpuRequired: cloudExecutionMode === 'gpu' ? true : cloudExecutionMode === 'prepare' || autoPrepareOnNoGpu ? false : undefined,
        policyType: effectiveCloudPolicyType,
        modelFamily: effectiveCloudPolicyType,
        ...buildRlinfRuntimeParams(),
        resourceIntent: 'auto',
        allowUnverifiedRuntime: false,
        gpuCount: cloudGpuCount,
        replicas: cloudReplicaCount,
        requestedGpuTotal,
        autodlRegionSign: cloudRegionSign || undefined,
        autodlDataCenters: cloudRegionSign || undefined,
        ...environmentParams,
        steps: trainSteps,
        device: trainDevice,
      },
    })
    const vlaPlan = extractVlaPlanPayload(data)
    const nextProvider = firstString((data as any)?.provider, toRecord((data as any)?.bridge).provider, vlaPlan.provider)
    if (nextProvider === 'autodl' || nextProvider === 'aliyun') {
      setCloudProvider(nextProvider)
    }
    if (vlaPlan?.workflow) {
      setCloudWorkflow(String(vlaPlan.workflow))
    }
    if (vlaPlan?.params?.modelFamily) {
      setPolicyType(String(vlaPlan.params.modelFamily))
    }
    if (Object.keys(vlaPlan).length > 0) {
      applyVlaPlanToForm(vlaPlan)
    }
  }

  useEffect(() => {
    if (
      trainMode !== 'cloud' ||
      activeCloudJob ||
      trainingPlanLoading ||
      trainingLoading ||
      trainPlan ||
      cloudIntent.trim() ||
      requiresRuntimeRebind ||
      !cloudTaskQueue.length
    ) {
      return
    }
    const nextTask = cloudTaskQueue[0]
    if (!nextTask || queuedPlanningRef.current === nextTask.id) return
    queuedPlanningRef.current = nextTask.id
    setCloudTaskQueue((current) => current.filter((item) => item.id !== nextTask.id))
    setCloudIntent(nextTask.text)
    window.setTimeout(() => {
      void planCloudTraining(nextTask.text).finally(() => {
        queuedPlanningRef.current = ''
      })
    }, 0)
  }, [
    activeCloudJob,
    cloudIntent,
    cloudTaskQueue,
    requiresRuntimeRebind,
    trainMode,
    trainPlan,
    trainingLoading,
    trainingPlanLoading,
  ])

  const openAdvancedOptions = (nextOpen = !showAdvancedCloudOptions) => {
    if (!nextOpen) setShowExpertCloudOptions(false)
    setShowAdvancedCloudOptions(nextOpen)
  }
  const focusLaunchSummaryField = (target: LaunchSummaryAction) => {
    setInlineConfirmEdit(current => current === target ? '' : target)
  }

  useEffect(() => {
    const handleEditTrainingGuidance = () => {
      focusLaunchSummaryField('intent')
    }
    window.addEventListener('evo-studio:edit-training-guidance', handleEditTrainingGuidance)
    return () => window.removeEventListener('evo-studio:edit-training-guidance', handleEditTrainingGuidance)
  }, [])
  const confirmationParams = {
    ...toRecord(displayedVlaPlan?.params),
    ...(aiConfiguredParams || {}),
    ...manualTrainingOverrides,
  }
  const confirmationDatasetSource = toRecord(confirmationParams.datasetSource)
  const confirmationModelSource = toRecord(confirmationParams.modelSource)
  const rlinfConfigName = firstString(
    confirmationParams.configName,
    confirmationParams.rlinfConfigName,
    confirmationParams.builtinTrainingProfile,
    toRecord(confirmationParams.rlinfConfig).configName,
  )
  const rawLaunchDatasetValue = benchmarkDatasetMismatch
    ? '未解析：ManiSkill 方案不能使用 LIBERO 数据'
    : firstString(
        selectedCloudDataset?.label,
        cloudSourceUri,
        confirmationDatasetSource.datasetId,
        confirmationDatasetSource.uri,
        confirmationParams.datasetPath,
        confirmationParams.dataPath,
        OFFICIAL_DATASET_SOURCES[cloudBenchmark]?.label,
        cloudBenchmark,
      )
  const rawLaunchModelValue = firstString(
    cloudModelUri,
    confirmationModelSource.uri,
    confirmationModelSource.checkpoint,
    confirmationModelSource.modelFamily,
    confirmationParams.modelFamily,
    confirmationParams.policyType,
    confirmationParams.checkpointPath,
    effectiveCloudPolicyType,
  )
  const launchModelFamily = firstString(
    confirmationModelSource.modelFamily,
    confirmationParams.modelFamily,
    confirmationParams.policyType,
    effectiveCloudPolicyType,
  )
  const launchModelValue = rlinfConfigName && isUnresolvedModelText(rawLaunchModelValue)
    ? `${displayModelFamily(launchModelFamily, rlinfConfigName)} · RLinf config: ${rlinfConfigName}`
    : looksLikeInternalPath(rawLaunchModelValue) && launchModelFamily
      ? `${displayModelFamily(launchModelFamily, rlinfConfigName)}（云端模型缓存已解析）`
      : userFacingSourceValue(rawLaunchModelValue, '云端模型缓存已解析')
  const launchSummaryRows: Array<{ label: string; value: string; action?: LaunchSummaryAction }> = [
    {
      label: '任务',
	      value: sanitizePlanText(firstString(
	        intentUnderstandingSummary,
	        displayedVlaPlan?.aiSummary,
	        cloudIntent,
	      )),
	      action: 'intent' as LaunchSummaryAction,
    },
    {
	      label: '数据',
	      value: userFacingSourceValue(rawLaunchDatasetValue, '云端数据缓存已解析'),
	      action: 'data' as LaunchSummaryAction,
    },
    {
	      label: '模型',
	      value: launchModelValue,
	      action: 'model' as LaunchSummaryAction,
    },
    {
      label: '资源',
      value: isExistingSshRuntime
        ? '后端已配置 SSH GPU 实例'
        : [
            selectedCloudSku ? String(selectedCloudSku.displayName || selectedCloudSku.skuId) : '平台自动匹配',
            selectedCloudImage ? String(selectedCloudImage.displayName || selectedCloudImage.imageId) : '',
            Number.isFinite(selectedHourlyCostCents) && selectedHourlyCostCents > 0
              ? `约 ¥${(selectedHourlyCostCents / 100).toFixed(2)}/小时`
	            : '',
		          ].filter(Boolean).join(' · '),
	      action: 'resource' as LaunchSummaryAction,
    },
    {
      label: '结果',
      value: firstString(
        confirmationParams.artifactPath,
        confirmationParams.artifactOutputPath,
        confirmationParams.outputDir,
      )
        ? '云端保存日志、指标和产物'
        : '保存日志、指标和产物',
    },
  ].filter(row => row.value)

  return (
    <div className="page-enter flex flex-col h-full overflow-y-auto">
      <div className="border-b border-bd/50 px-6 py-4 bg-sf">
        <h2 className="text-xl font-bold tracking-tight">{t('trainingCenter')}</h2>
      </div>

      <div className="flex-1 p-6 grid grid-cols-2 gap-6 items-start max-[1100px]:grid-cols-1">
        <section className="bg-sf rounded-xl p-5 shadow-card shadow-inset-yl">
          <div className="flex items-start justify-between gap-4 mb-4">
            <div>
              <h3 className="text-sm font-bold text-tx uppercase tracking-wide">
                {trainMode === 'cloud' ? 'Evo Studio 实验总控' : t('training')}
              </h3>
              <p className="mt-1 text-xs text-tx3 leading-relaxed">
                {trainMode === 'cloud'
                  ? '说出目标，Evo Studio 会把具身智能实验从数据准备、模型配置、云端运行到评测报告串起来；涉及费用、换机器或私有资源时先确认。'
                  : '数据负责训练样本；Benchmark 是独立的训练/评测环境，主要用于 RL 后训练和评测。'}
              </p>
            </div>
            <div className="shrink-0 flex rounded-lg border border-bd bg-bg p-1">
              {(['cloud', 'local'] as const).map(mode => (
                <button
                  key={mode}
                  onClick={() => setTrainMode(mode)}
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors ${
                    trainMode === mode ? 'bg-ac text-white shadow-glow-ac' : 'text-tx3 hover:text-tx hover:bg-sf2'
                  }`}
                >
                  {mode === 'cloud' ? '云训练' : '本地训练'}
                </button>
              ))}
            </div>
          </div>

          {trainMode === 'local' ? (
            <select
              value={trainDataset}
              onChange={(e) => setTrainDataset(e.target.value)}
              className="w-full bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm mb-3
                focus:outline-none focus:border-ac"
            >
              <option value="">{t('selectDataset')}</option>
              {runtimeDatasets.map(d => (
                <option key={d.id} value={d.runtime!.name}>{d.label}</option>
              ))}
            </select>
          ) : (
            <div className="space-y-3 mb-4">
              <div className="rounded-xl border border-ac/25 bg-bg p-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold text-tx">实验总控</div>
                    <div className="mt-1 text-xs text-tx3">一句话下达目标；需要确认时才打断你。</div>
                  </div>
                  <details className="relative shrink-0 text-xs text-tx3">
                    <summary className="cursor-pointer select-none rounded-md border border-bd px-2.5 py-1 font-semibold text-tx2 hover:border-ac hover:text-ac">
                      详情
                    </summary>
                    <div className="absolute right-0 z-10 mt-2 w-[min(360px,calc(100vw-3rem))] rounded-lg border border-bd bg-sf p-3 shadow-card">
                      <div className="text-xs font-semibold text-tx2">
                        {activeCloudJob
                          ? '任务运行中'
                          : trainingPlanLoading
                            ? '正在理解指令'
                            : trainPlan
                              ? '等待确认启动'
                              : '空闲'}
                      </div>
                      <div className="mt-1 text-[11px] text-tx3 leading-relaxed">
                        接入的大模型负责理解目标；后端工具负责执行、检查和续跑。
                      </div>
                      <details className="mt-2 rounded-md border border-bd/50 bg-bg px-2.5 py-2">
                        <summary className="cursor-pointer select-none text-[11px] font-semibold text-tx2">能力目录</summary>
                        <div className="mt-1 text-[11px] text-tx3">
                          {rlinfCatalogSummary.configured
                            ? `RLinf：${rlinfCatalogSummary.count} 个 recipe${rlinfCatalogSummary.benchmarks.length ? ` · ${rlinfCatalogSummary.benchmarks.slice(0, 4).join(' / ')}` : ''}`
                            : 'RLinf：未连接'}
                        </div>
                      </details>
                    </div>
                  </details>
                </div>

                <div className="mt-4 grid gap-3">
                  {!activeCloudJob && cloudIntent.trim() && (
                    <div className="flex justify-end">
                      <div className="max-w-[86%] rounded-2xl rounded-tr-md bg-ac px-3 py-2 text-sm text-white leading-relaxed whitespace-pre-wrap">
                        {cloudIntent.trim()}
                      </div>
                    </div>
                  )}

                  {!activeCloudJob && (trainingPlanLoading || trainPlanMessage || trainPlan || aiProviderConfigured === false || cloudBridgeChecking || !cloudBridgeConnected || !cloudRuntimeReady) && (
                    <div className="flex items-start gap-2">
                      <div className="grid h-7 w-7 shrink-0 place-items-center rounded-full bg-ac/10 text-xs font-bold text-ac">AI</div>
                      <div className="max-w-[94%] rounded-2xl rounded-tl-md border border-bd/70 bg-sf px-3 py-2">
                        {trainingPlanLoading && (
                          <div className="text-sm font-semibold text-tx">正在理解需求并检查云端资源...</div>
                        )}
                        {aiProviderConfigured === false && (
                          <div className="mt-1 rounded-md border border-yl/40 bg-yl/10 px-3 py-2 text-xs text-tx2 leading-relaxed">
                            未接入可用大模型；不会伪装成智能规划。
                          </div>
                        )}
                        {cloudBridgeStatus?.enabled === false && (
                          <div className="mt-1 rounded-md border border-yl/40 bg-yl/10 px-3 py-2 text-xs text-tx2 leading-relaxed">
                            <div className="font-semibold text-tx">云训练后台未连接</div>
                            <div className="mt-1">需要先恢复 EVO_Train 桥接，再提交云端任务。</div>
                          </div>
                        )}
                        {cloudBridgeChecking && (
                          <div className="mt-1 rounded-md border border-bd/60 bg-bg/70 px-3 py-2 text-xs text-tx2 leading-relaxed">
                            正在检查云端实例连接...
                          </div>
                        )}
                        {cloudBridgeConnected && !cloudRuntimeReady && (
                          <div className="mt-1 rounded-md border border-yl/40 bg-yl/10 px-3 py-2 text-xs text-tx2 leading-relaxed">
                            <div className="font-semibold text-tx">云端实例未就绪，请重新连接</div>
                            {cloudBridgeStatus?.runtimeEndpoint && (
                              <div className="mt-1 font-mono text-[11px] text-tx3">
                                当前绑定：{cloudBridgeStatus.runtimeEndpoint}
                              </div>
                            )}
                            <div className="mt-1">
                              {userFacingCloudRuntimeWarning(cloudBridgeStatus?.configurationWarnings?.[0] || cloudBridgeStatus?.message)}
                            </div>
                          </div>
                        )}
                        {canRebindSshRuntime && (!cloudBridgeConnected || !cloudRuntimeReady) && (
                          <div ref={sshBindSectionRef} className="mt-2 rounded-md border border-yl/30 bg-bg/70 px-3 py-2 text-xs text-tx3 scroll-mt-4">
                            <div className="font-semibold text-tx2">连接当前云端实例</div>
                            <div className="mt-1 leading-relaxed">
                              粘贴 AutoDL / SeetaCloud 的 SSH 命令和密码；保存后平台会连接这台实例。
                            </div>
                            <div className="mt-3 grid gap-3">
                              <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                                SSH 命令
                                <input
                                  value={sshBindCommand}
                                  onChange={(e) => setSshBindCommand(e.target.value)}
                                  placeholder="ssh -p 42552 root@connect.cqa1.seetacloud.com"
                                  className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                                />
                              </label>
                              <div className="grid grid-cols-2 gap-3 max-[760px]:grid-cols-1">
                                <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                                  SSH 密码
                                  <input
                                    type="password"
                                    value={sshBindPassword}
                                    onChange={(e) => setSshBindPassword(e.target.value)}
                                    placeholder="只发送给本机后端，不进入训练参数"
                                    className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                                  />
                                </label>
                                <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                                  私钥路径（可选）
                                  <input
                                    value={sshBindKeyPath}
                                    onChange={(e) => setSshBindKeyPath(e.target.value)}
                                    placeholder="~/.ssh/id_rsa"
                                    className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                                  />
                                </label>
                              </div>
                              <div className="flex flex-wrap items-center gap-3">
                                <button
                                  type="button"
                                  disabled={sshBindLoading}
                                  onClick={() => { void bindSshRuntime() }}
                                  className="px-3 py-2 rounded-md bg-ac text-white text-xs font-semibold hover:bg-ac2 disabled:opacity-40 disabled:cursor-not-allowed"
                                >
                                  {sshBindLoading ? '正在等待 SSH...' : '保存并连接实例'}
                                </button>
                                {cloudBridgeStatus?.runtimeEndpoint && (
                                  <button
                                    type="button"
                                    disabled={sshBindLoading}
                                    onClick={() => { void clearSshRuntimeBinding() }}
                                    className="px-3 py-2 rounded-md border border-bd text-xs font-semibold text-tx2 hover:border-rd hover:text-rd disabled:opacity-40 disabled:cursor-not-allowed"
                                  >
                                    清除旧绑定
                                  </button>
                                )}
                                {sshBindMessage && <span className="text-xs text-tx3">{sshBindMessage}</span>}
                              </div>
                            </div>
                          </div>
                        )}
                        {Boolean(trainPlanMessage || trainPlan) && (
                          <div className="mt-2">
                            {trainPlanMessage && !aiPlannerUnavailable && <div className="text-sm font-semibold text-tx">{sanitizePlanText(trainPlanMessage)}</div>}
                            {displayedPlannerSource === 'llm' && (
                              <div className={`mt-2 rounded-md border px-3 py-2 text-xs font-semibold ${
                                'border-gn/30 bg-gn/10 text-tx2'
                              }`}>
                                {`接入的大模型已理解指令${displayedPlannerModel ? ` · ${displayedPlannerModel}` : ''}`}
                              </div>
                            )}
                            {aiPlannerUnavailable && (
                              <div className="mt-2 rounded-md border border-yl/40 bg-yl/10 px-3 py-2">
                                <div className="text-xs font-semibold text-tx">{aiPlannerUnavailableLabel}</div>
                                <div className="mt-1 text-xs text-tx3 leading-relaxed">
                                  {aiPlannerUnavailableText}
                                </div>
                                <button
                                  type="button"
                                  disabled={trainingPlanLoading || !cloudIntent.trim()}
                                  onClick={() => { void planCloudTraining() }}
                                  className="mt-2 px-3 py-1.5 rounded-md bg-ac text-white text-xs font-semibold hover:bg-ac2 disabled:opacity-40 disabled:cursor-not-allowed"
                                >
                                  {trainingPlanLoading ? '重试中...' : '重试'}
                                </button>
                              </div>
                            )}
                            {displayedPlannerSource === 'llm' && intentUnderstandingSummary && (
                              <div className="mt-2 rounded-md border border-gn/25 bg-gn/5 px-3 py-2 text-xs text-tx2 leading-relaxed">
                                我准备这样做：{sanitizePlanText(intentUnderstandingSummary)}
                              </div>
                            )}
                            {clarifyingQuestions.length > 0 && (
                              <div className="mt-2 rounded-md border border-ac/25 bg-ac/5 px-3 py-2">
                                <div className="text-xs font-semibold text-tx">需要你确认</div>
                                <div className="mt-1 grid gap-1.5 text-xs text-tx2 leading-relaxed">
                                  {clarifyingQuestions.slice(0, 3).map((question, index) => (
                                    <div key={`${question}-${index}`}>{index + 1}. {sanitizePlanText(question)}</div>
                                  ))}
                                </div>
                              </div>
                            )}
                            {!activeCloudJob && !requiresRuntimeRebind && !aiPlannerUnavailable && launchSummaryRows.length > 0 && (
                              <div className="mt-2 rounded-md border border-ac/25 bg-bg px-3 py-2">
                                <div className="flex items-center justify-between gap-3">
                                  <div className="text-xs font-semibold text-tx">启动前确认</div>
                                  <div className="text-[11px] text-tx3">实例就绪后点主按钮提交</div>
                                </div>
                                <div className="mt-2 grid grid-cols-2 gap-2 max-[680px]:grid-cols-1">
                                  {launchSummaryRows.map(row => (
                                    <div key={row.label} className="grid grid-cols-[42px_1fr_auto] items-center gap-2 rounded-md border border-bd/50 bg-sf px-2.5 py-2">
                                      <div className="text-[11px] font-semibold text-tx3">{row.label}</div>
                                      <div className="text-xs text-tx2 truncate" title={row.value}>{row.value}</div>
                                      {row.action && (
                                        <button
                                          type="button"
                                          onClick={() => focusLaunchSummaryField(row.action!)}
                                          className="rounded px-2 py-1 text-[11px] font-semibold text-ac hover:bg-ac/10 transition-colors"
                                        >
                                          {inlineConfirmEdit === row.action ? '收起' : '编辑'}
                                        </button>
                                      )}
                                      {row.action && inlineConfirmEdit === row.action && (
                                        <div className="col-span-3 mt-1 rounded-md border border-bd/50 bg-bg px-2.5 py-2">
                                          {row.action === 'intent' && (
                                            <textarea
                                              value={cloudIntent}
                                              onChange={(e) => {
                                                setCloudIntent(e.target.value)
                                                invalidateCloudPlan()
                                              }}
                                              rows={2}
                                              placeholder="补充或改写目标"
                                              className="w-full resize-none bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-xs leading-relaxed focus:outline-none focus:border-ac"
                                            />
                                          )}
                                          {row.action === 'data' && (
                                            <input
                                              value={cloudSourceUri}
                                              onChange={(e) => {
                                                const nextValue = e.target.value
                                                setCloudSourceUri(nextValue)
                                                invalidateCloudPlan()
                                                if (nextValue.trim()) {
                                                  setCloudSourceKind(inferCloudSourceKindFromUri(nextValue))
                                                }
                                              }}
                                              placeholder="HuggingFace / ModelScope 地址，留空由 AI 自动匹配"
                                              className="w-full bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-xs font-mono focus:outline-none focus:border-ac"
                                            />
                                          )}
                                          {row.action === 'model' && (
                                            <div className="grid grid-cols-[1fr_auto] gap-2 max-[680px]:grid-cols-1">
                                              <input
                                                value={cloudModelUri}
                                                onChange={(e) => {
                                                  const nextValue = e.target.value
                                                  setCloudModelUri(nextValue)
                                                  invalidateCloudPlan()
                                                  setCloudModelInitSource(nextValue.trim() ? 'external' : 'auto')
                                                }}
                                                placeholder="模型仓库或 checkpoint 地址，留空则由总控判断"
                                                className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-xs font-mono focus:outline-none focus:border-ac"
                                              />
                                              <button
                                                type="button"
                                                onClick={() => {
                                                  setCloudModelUri('')
                                                  setCloudModelInitSource('auto')
                                                  invalidateCloudPlan()
                                                }}
                                                className="rounded-lg border border-bd px-3 py-2 text-xs font-semibold text-tx2 hover:border-ac hover:text-ac"
                                              >
                                                自动判断
                                              </button>
                                            </div>
                                          )}
                                          {row.action === 'resource' && (
                                            <div className="flex flex-wrap items-center gap-2">
                                              {([
                                                ['auto', '总控自动'],
                                                ['prepare', '无卡准备'],
                                                ['gpu', 'GPU 运行'],
                                              ] as const).map(([mode, label]) => (
                                                <button
                                                  key={mode}
                                                  type="button"
                                                  onClick={() => setCloudExecutionMode(mode)}
                                                  className={`rounded-md border px-2.5 py-1.5 text-xs font-semibold transition-colors ${
                                                    cloudExecutionMode === mode
                                                      ? 'border-ac bg-ac/10 text-ac'
                                                      : 'border-bd bg-sf text-tx2 hover:border-ac/60 hover:text-ac'
                                                  }`}
                                                >
                                                  {label}
                                                </button>
                                              ))}
                                              <button
                                                type="button"
                                                onClick={() => {
                                                  setShowSshRuntimeBind(true)
                                                  setInlineConfirmEdit('')
                                                }}
                                                className="rounded-md border border-bd px-2.5 py-1.5 text-xs font-semibold text-tx2 hover:border-ac hover:text-ac"
                                              >
                                                重新绑定实例
                                              </button>
                                            </div>
                                          )}
                                        </div>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                            {!aiPlannerUnavailable && (
                              <details className="mt-2 rounded-md border border-bd/50 bg-bg px-3 py-2 text-xs text-tx3">
                                <summary className="cursor-pointer select-none font-semibold text-tx2">执行详情</summary>
                                {Array.isArray(displayedVlaPlan?.planSteps) && displayedVlaPlan.planSteps.length > 0 && (
                                  <div className="mt-2 grid gap-1.5">
                                    {displayedVlaPlan.planSteps.map((step: string, index: number) => (
                                      <div key={`${sanitizePlanText(step)}-${index}`} className="flex gap-2 text-xs text-tx2 leading-relaxed">
                                        <span className="shrink-0 w-5 h-5 rounded-full bg-ac/10 text-ac grid place-items-center font-semibold">
                                          {index + 1}
                                        </span>
                                        <span>{sanitizePlanText(step)}</span>
                                      </div>
                                    ))}
                                  </div>
                                )}
                                {Array.isArray(displayedVlaPlan?.evaluationPlan) && displayedVlaPlan.evaluationPlan.length > 0 && (
                                  <div className="mt-2 text-xs text-tx2 leading-relaxed">
                                    <span className="font-semibold text-tx">评测输出：</span>
                                    {displayedVlaPlan.evaluationPlan.map((item: string) => sanitizePlanText(item)).join('；')}
                                  </div>
                                )}
                                {Array.isArray(displayedVlaPlan?.resourceHints) && displayedVlaPlan.resourceHints.length > 0 && (
                                  <div className="mt-2 text-xs text-tx2 leading-relaxed">
                                    <span className="font-semibold text-tx">资源备注：</span>
                                    {displayedVlaPlan.resourceHints.map((item: string) => sanitizePlanText(item)).join('；')}
                                  </div>
                                )}
                              </details>
                            )}
                            {trainPlan && !aiPlannerUnavailable && startBlockers.length > 0 && (
                              <div className="mt-2 rounded-md border border-yl/40 bg-yl/10 px-3 py-2 text-xs text-tx2 leading-relaxed">
                                还不能提交：{startBlockers[0]}
                              </div>
                            )}
                            {trainPlan && showExpertCloudOptions && (
                              <details className="mt-2 rounded-md border border-bd/50 bg-bg px-3 py-2 text-xs text-tx3">
                                <summary className="cursor-pointer select-none font-semibold text-tx2">开发调试：查看后台原始参数</summary>
                                <pre className="mt-2 max-h-[220px] overflow-auto whitespace-pre-wrap break-all text-[11px] leading-relaxed text-tx2 font-mono">
                                  {JSON.stringify((trainPlan as any).vlaPlan || trainPlan, null, 2)}
                                </pre>
                                {trainJobHistory.length > 0 && (
                                  <details className="mt-2 rounded-md border border-bd/50 bg-sf px-2.5 py-2">
                                    <summary className="cursor-pointer select-none font-semibold text-tx2">
                                      后台任务记忆（{trainJobHistory.length}）
                                    </summary>
                                    <div className="mt-2 grid gap-2">
                                      {trainJobHistory.slice(0, 6).map(item => (
                                        <div key={item.jobId} className="rounded-md border border-bd/50 bg-bg px-2.5 py-2">
                                          <div className="flex items-center justify-between gap-3">
                                            <span className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold ${
                                              /fail|error/i.test(item.status)
                                                ? 'border-rd/40 bg-rd/10 text-rd'
                                                : /stop/i.test(item.status)
                                                  ? 'border-yl/40 bg-yl/10 text-yl'
                                                  : 'border-bd bg-sf text-tx2'
                                            }`}>
                                              {/fail|error/i.test(item.status)
                                                ? '失败'
                                                : /stop/i.test(item.status)
                                                  ? '已停止'
                                                  : item.status || '已记录'}
                                            </span>
                                            <span className="text-[11px] text-tx3">{new Date(item.updatedAt).toLocaleString()}</span>
                                          </div>
                                          <div className="mt-1 truncate font-mono text-[11px] text-tx3" title={item.jobId}>{item.jobId}</div>
                                        </div>
                                      ))}
                                    </div>
                                  </details>
                                )}
                              </details>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                <div
                  onClickCapture={(event) => {
                    const target = event.target
                    if (!(target instanceof HTMLElement)) return
                    if (!target.closest('[data-cloud-start-button="true"]')) return
                    event.preventDefault()
                    handleCloudPrimaryAction()
                  }}
                >
                  <CloudIntentPanel
                    intent={cloudIntent}
                    intentRef={cloudIntentRef}
                    queue={cloudTaskQueue}
                    onIntentChange={(nextValue) => {
                      setCloudIntent(nextValue)
                      invalidateCloudPlan()
                    }}
                    onRemoveQueuedTask={removeQueuedCloudTask}
                    sourcePanel={(
                      <CloudSourcePanel
                        open={showAdvancedCloudOptions}
                        sourceUri={cloudSourceUri}
                        modelUri={cloudModelUri}
                        sourceInputRef={cloudSourceUriRef}
                        modelInputRef={cloudModelUriRef}
                        onToggle={openAdvancedOptions}
                        onSourceUriChange={(nextValue) => {
                          setCloudSourceUri(nextValue)
                          invalidateCloudPlan()
                          if (nextValue.trim()) {
                            setCloudSourceKind(inferCloudSourceKindFromUri(nextValue))
                          } else if (cloudSourceKind === 'public_reference' || cloudSourceKind === 'user_object_storage') {
                            setCloudSourceKind('official_reference')
                          }
                        }}
                        onModelUriChange={(nextValue) => {
                          setCloudModelUri(nextValue)
                          invalidateCloudPlan()
                          if (nextValue.trim()) {
                            setCloudModelInitSource('external')
                          } else if (cloudModelInitSource === 'external') {
                            setCloudModelInitSource('auto')
                          }
                        }}
                      />
                    )}
                    actions={(
                      <>
                        {cloudCheckLoading && (
                          <button
                            type="button"
                            onClick={cancelCloudChecks}
                            className="px-3 py-2 rounded-lg border border-bd bg-sf text-sm font-semibold text-tx hover:border-rd/60 hover:text-rd transition-all active:scale-[0.97]"
                          >
                            取消
                          </button>
                        )}
                        {renderCloudStartButton('px-4 py-2 rounded-lg')}
                      </>
                    )}
                    helperText={
                      activeCloudJob
                        ? '当前云任务运行中；继续发送会进入待处理队列，等当前任务结束后依次处理。'
                        : cloudRestartReady
                          ? (restartableCloudJobId
                              ? '当前实例已恢复；点击“继续运行”会复用上次任务参数。'
                              : '当前实例已恢复；点击“重新生成方案”后即可启动新任务。')
                        : cloudBridgeChecking
                          ? '正在检查云端实例连接。'
                        : requiresRuntimeRebind
                          ? '先连接云端实例。'
                          : '发送后总控会自动判断准备、运行和评测步骤；忙时会排队。'
                    }
                  />
                </div>
                {trainJobMessage && !activeCloudJob && !requiresRuntimeRebind && (
                  <div
                    role="status"
                    aria-live="polite"
                    className="mt-2 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-bd/60 bg-bg px-3 py-2 text-xs text-tx2"
                  >
                    <span>{sanitizePlanText(summarizeTrainJobMessage(trainJobMessage))}</span>
                    {cloudRestartReady && !aiPlannerUnavailable && (
                      <button
                        type="button"
                        disabled={cloudStartDisabled}
                        onClick={handleCloudPrimaryAction}
                        className="rounded-md bg-ac px-2.5 py-1.5 text-xs font-semibold text-white hover:bg-ac2 disabled:opacity-40 disabled:cursor-not-allowed"
                      >
                        {restartableCloudJobId ? '继续运行' : trainPlan ? '重新启动任务' : '重新生成方案'}
                      </button>
                    )}
                  </div>
                )}
              </div>

              <CloudProviderPanel
                statusKnown={Boolean(cloudBridgeStatus)}
                connected={cloudBridgeConnected}
                ready={cloudRuntimeReady}
                managed={isManagedComputePool}
                canRebind={canRebindSshRuntime}
                canShowDebug={canShowInternalTrainingDebug}
                showDebug={showExpertCloudOptions}
                readySkuCount={readySkuCount}
                totalSkuCount={totalSkuCount}
                readyImageCount={readyImageCount}
                totalImageCount={totalImageCount}
                runtimeEndpoint={cloudBridgeStatus?.runtimeEndpoint}
                onRebind={() => {
                  setShowSshRuntimeBind(true)
                  window.setTimeout(() => {
                    sshBindSectionRef.current?.scrollIntoView({ block: 'start', behavior: 'smooth' })
                  }, 80)
                }}
                onRefresh={() => {
                  void loadCloudBridgeStatus()
                  void loadCloudResources(cloudProvider)
                }}
                onToggleDebug={() => setShowExpertCloudOptions(value => !value)}
              />

              {canRebindSshRuntime && (showSshRuntimeBind || !cloudBridgeConnected || !cloudRuntimeReady) && (
                <details
                  ref={sshBindSectionRef}
                  open={showSshRuntimeBind}
                  onToggle={(event) => setShowSshRuntimeBind(event.currentTarget.open)}
                  className="rounded-lg border border-bd/60 bg-bg px-3 py-2 text-xs text-tx2 scroll-mt-4"
                >
	                  <summary className="cursor-pointer select-none font-semibold text-tx hover:text-ac">
	                    更换云端实例
	                  </summary>
	                  <div className="mt-2 text-tx3 leading-relaxed">
	                    换了新的 AutoDL / SeetaCloud 实例时，在这里粘贴新的 SSH 命令和密码或私钥路径；保存后平台会连接这台实例。
                  </div>
                  <div className="mt-3 grid gap-3">
                    <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                      SSH 命令
                      <input
                        value={sshBindCommand}
                        onChange={(e) => setSshBindCommand(e.target.value)}
                        placeholder="ssh -p 42552 root@connect.cqa1.seetacloud.com"
                        className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                      />
                    </label>
                    <div className="grid grid-cols-2 gap-3 max-[760px]:grid-cols-1">
                      <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                        SSH 密码
                        <input
                          type="password"
                          value={sshBindPassword}
                          onChange={(e) => setSshBindPassword(e.target.value)}
                          placeholder="只发送给本机后端，不进入训练参数"
                          className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                        />
                      </label>
                      <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                        私钥路径（可选）
                        <input
                          value={sshBindKeyPath}
                          onChange={(e) => setSshBindKeyPath(e.target.value)}
                                    placeholder="~/.ssh/id_rsa"
                          className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                        />
                      </label>
                    </div>
                    <div className="flex flex-wrap items-center gap-3">
                      <button
                        type="button"
                        disabled={sshBindLoading}
                        onClick={() => { void bindSshRuntime() }}
                        className="px-3 py-2 rounded-md bg-ac text-white text-xs font-semibold hover:bg-ac2 disabled:opacity-40 disabled:cursor-not-allowed"
                      >
	                        {sshBindLoading ? '正在等待 SSH...' : '保存并连接实例'}
                      </button>
                      {cloudBridgeStatus?.runtimeEndpoint && (
                        <button
                          type="button"
                          disabled={sshBindLoading}
                          onClick={() => { void clearSshRuntimeBinding() }}
                          className="px-3 py-2 rounded-md border border-bd text-xs font-semibold text-tx2 hover:border-rd hover:text-rd disabled:opacity-40 disabled:cursor-not-allowed"
                        >
                          清除旧绑定
                        </button>
                      )}
                      {sshBindMessage && <span className="text-xs text-tx3">{sshBindMessage}</span>}
                    </div>
                  </div>
                </details>
              )}

              {activeCloudJob && (
                <div className="rounded-lg border border-ac/25 bg-ac/5 px-3 py-2">
                  <div className="flex items-center justify-between gap-3 max-[640px]:flex-col max-[640px]:items-start">
                    <div className="min-w-0">
                      <div className="text-xs font-semibold text-tx">当前运行任务</div>
                      <div className="mt-1 text-[11px] text-tx3 truncate" title={currentTrainJobId}>
                        {currentTrainJobId}
                      </div>
                    </div>
                    <div className="flex shrink-0 flex-wrap items-center gap-2">
                      <div className="rounded-full border border-ac/35 bg-ac/10 px-2.5 py-1 text-[11px] font-semibold text-ac">
                        {currentCloudJobStatus}
                      </div>
                      {renderCloudSafetyControls('')}
                    </div>
                  </div>
                </div>
              )}

              {canShowInternalTrainingDebug && showExpertCloudOptions && (
                <>
              <div ref={advancedCloudOptionsRef} className="rounded-lg border border-ac/30 bg-ac/5 p-3 scroll-mt-4">
                <div className="flex items-start justify-between gap-3 max-[760px]:flex-col">
                  <div>
                    <div className="text-sm font-semibold text-tx">高级设置</div>
                    <div className="mt-1 text-xs text-tx3 leading-relaxed">
                      一般不用填；只在你要覆盖 AI 方案时改这里。确认前不会提交任务。
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => { void planCloudTraining() }}
                    disabled={trainingPlanLoading || !cloudIntent.trim()}
                    className="px-3 py-1.5 rounded-md bg-ac text-white text-xs font-semibold hover:bg-ac/90 disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    重新检查
                  </button>
                </div>
                <div className="mt-3 grid grid-cols-4 gap-3 max-[1100px]:grid-cols-2 max-[640px]:grid-cols-1">
                  <label className="col-span-2 flex flex-col gap-1 text-2xs text-tx3 font-mono max-[1100px]:col-span-1">
                    补充要求
                    <textarea
                      value={cloudExecutionNotes}
                      onChange={(e) => {
                        setCloudExecutionNotes(e.target.value)
                        invalidateCloudPlan()
                      }}
                      rows={2}
                      placeholder="例如：只跑 smoke test、每个任务 2 次、失败后在同一实例继续修复。"
                      className="min-h-[64px] resize-y bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                    />
                  </label>
                  <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                    次数 / 步数
                    <input
                      type="number"
                      min={1}
                      value={trainSteps}
                      onChange={(e) => setTrainSteps(Math.max(1, Number(e.target.value) || 1))}
                      className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                    />
                  </label>
                  <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                    learning rate
                    <input
                      value={cloudLearningRate}
                      onChange={(e) => setCloudLearningRate(e.target.value)}
                      placeholder="留空则使用 recipe 默认"
                      className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                    />
                  </label>
                  <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                    batch size
                    <input
                      value={cloudBatchSize}
                      onChange={(e) => setCloudBatchSize(e.target.value)}
                      placeholder="留空则使用 recipe 默认"
                      className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                    />
                  </label>
                </div>
                <details className="mt-3 rounded-md border border-bd/50 bg-bg px-3 py-2 text-xs text-tx3">
                  <summary className="cursor-pointer select-none font-semibold text-tx2">更多内部参数</summary>
                  <div className="mt-3 grid grid-cols-4 gap-3 max-[1100px]:grid-cols-2 max-[640px]:grid-cols-1">
                    <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                      执行方式
                      <select
                        value={cloudWorkflow}
                        onChange={(e) => setCloudWorkflow(e.target.value)}
                        className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                      >
                        {TRAINING_WORKFLOWS.map(workflow => (
                          <option key={workflow.id} value={workflow.id}>{workflow.label}</option>
                        ))}
                      </select>
                    </label>
                    <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                      模型 / 策略标识
                      <input
                        value={policyType}
                        onChange={(e) => setPolicyType(e.target.value)}
                        placeholder="例如 openvla、pi0、act、custom"
                        className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                      />
                    </label>
                    <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                      运行设备
                      <select
                        value={trainDevice}
                        onChange={(e) => setTrainDevice(e.target.value)}
                        className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                      >
                        <option value="cuda">cuda</option>
                        <option value="cpu">cpu</option>
                      </select>
                    </label>
                    <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                      epochs
                      <input
                        value={cloudEpochs}
                        onChange={(e) => setCloudEpochs(e.target.value)}
                        placeholder="可选"
                        className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                      />
                    </label>
                    <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                      warmup steps
                      <input
                        value={cloudWarmupSteps}
                        onChange={(e) => setCloudWarmupSteps(e.target.value)}
                        placeholder="可选"
                        className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                      />
                    </label>
                    <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                      梯度累积
                      <input
                        value={cloudGradientAccumulationSteps}
                        onChange={(e) => setCloudGradientAccumulationSteps(e.target.value)}
                        placeholder="可选"
                        className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                      />
                    </label>
                    <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                      LoRA rank
                      <input
                        value={cloudLoraRank}
                        onChange={(e) => setCloudLoraRank(e.target.value)}
                        placeholder="可选"
                        className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                      />
                    </label>
                  </div>
                </details>
              </div>
              <div className="rounded-lg border border-bd/50 bg-bg p-3">
                <div className="mb-3">
                  <div className="text-sm font-semibold text-tx">调试：资源覆盖</div>
                  <div className="mt-1 text-xs text-tx3 leading-relaxed">
                    正常情况下由 Agent 和后端自动选择。这里仅用于开发调试或强制指定某个机型/镜像；云厂商 token、实例创建、SSH 和镜像 UUID 由管理员在后端维护。
                  </div>
                </div>
                {cloudBridgeStatus?.enabled && (
                  <div className={`mb-3 rounded-md border px-3 py-2 text-xs leading-relaxed ${
                    isManagedComputePool
                      ? 'border-gn/35 bg-gn/10 text-tx2'
                      : 'border-yl/40 bg-yl/10 text-tx2'
                  }`}>
                    <div className="font-semibold text-tx">
                      {isManagedComputePool
                        ? '托管算力池已连接'
                        : isExistingDebugInstance
                          ? '当前连接的是已有调试实例'
                          : '云训练后端已连接'}
                    </div>
                    <div className="mt-1 text-tx3">
                      {isManagedComputePool
                        ? '用户可以从平台已开放、已计价、已验证的机型和镜像中选择；AI 会按任务需求推荐。'
                        : isExistingDebugInstance
                          ? '这是开发/内测模式：后端会复用管理员已配置的机器，不代表已经开放 AutoDL 全量自动创建资源池。'
                          : cloudBridgeStatus.message}
                    </div>
                    {(cloudCatalogSummary.readySkuCount !== undefined || cloudCatalogSummary.readyImageCount !== undefined) && (
                      <div className="mt-1 text-tx3">
                        可用机型 {String(cloudCatalogSummary.readySkuCount ?? 0)} / {String(cloudCatalogSummary.skuCount ?? 0)}，
                        可用镜像 {String(cloudCatalogSummary.readyImageCount ?? 0)} / {String(cloudCatalogSummary.imageCount ?? 0)}
                      </div>
                    )}
                  </div>
                )}
                <div className="grid grid-cols-6 gap-3 max-[1380px]:grid-cols-3 max-[760px]:grid-cols-1">
                  <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                    训练平台
                    <select
                      value={cloudProvider}
                      onChange={(e) => {
                        setCloudProvider(e.target.value as 'autodl' | 'aliyun')
                        invalidateCloudPlan()
                      }}
                      className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                    >
                      <option value="autodl">AutoDL 托管算力</option>
                      <option value="aliyun">阿里云 PAI 托管算力</option>
                    </select>
                  </label>
                  <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                    地区
                    <select
                      value={cloudRegionSign}
                      onChange={(e) => {
                        setCloudRegionSign(e.target.value)
                        setCloudSkuId('')
                        invalidateCloudPlan()
                      }}
                      className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                    >
                      <option value="">平台自动择区</option>
                      {regionCatalogRows.map((region) => (
                        <option key={region.regionSign} value={region.regionSign}>
                          {region.label}{region.availableGpuCount ? ` · 空闲 ${region.availableGpuCount} 张` : ''}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                    机型
                    <select
                      value={selectedSkuFamilyKey}
                      onChange={(e) => {
                        const familyKey = e.target.value
                        const option = machineOptions.find(item => item.key === familyKey)
                        const exactSku = regionFilteredCloudSkus.find(sku => (
                          skuFamilyKey(sku) === familyKey
                          && Number(sku.gpuCount || 1) === cloudGpuCount
                          && sku.stockStatus !== 'sold_out'
                          && skuAvailableInRegion(sku, cloudRegionSign)
                        ))
                        setCloudSkuId(String(exactSku?.skuId || option?.sku?.skuId || ''))
                        invalidateCloudPlan()
                      }}
                      className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                    >
                      <option value="">平台推荐</option>
                      {machineOptions.map((machine) => {
                        const price = machine.perGpuHourlyCents ? ` · ¥${(machine.perGpuHourlyCents / 100).toFixed(2)}/卡/小时` : ''
                        const stockText = machine.availableGpuCount > 0 ? ` · 空闲 ${machine.availableGpuCount} 张` : ''
                        const regionSummary = cloudRegionSign ? autodlRegionLabel(cloudRegionSign) : skuRegionSummary(machine.sku)
                        const regionText = regionSummary ? ` · ${regionSummary}` : ''
                        const stock = machine.stockStatus === 'sold_out'
                          ? ' · 暂无库存'
                          : machine.stockStatus === 'available'
                            ? stockText || ' · 有库存'
                            : ' · 库存待确认'
                        const ready = machine.hasReadySku ? '' : ' · 未就绪'
                        return (
                          <option key={machine.key} value={machine.key} disabled={machine.stockStatus === 'sold_out' || (Boolean(cloudRegionSign) && machine.availableGpuCount <= 0)}>
                            {machine.label}{price}{stock}{regionText}{ready}
                          </option>
                        )
                      })}
                    </select>
                  </label>
                  <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                    镜像
                    <select
                      value={cloudImageId}
                      onChange={(e) => {
                        setCloudImageId(e.target.value)
                        invalidateCloudPlan()
                      }}
                      className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                    >
                      <option value="">平台推荐</option>
                      {(cloudResourceCatalog?.images || []).map((image) => {
                        const id = String(image.imageId || '')
                        if (!id) return null
                        const cuda = image.cudaVFrom ? ` · CUDA ${image.cudaVFrom}` : ''
                        const source = String(image.source || '')
	                        const sourceLabel = source === 'autodl_public_base'
	                          ? ' · 官方'
	                          : source === 'autodl_community_configured'
	                            ? ' · 社区/团队'
	                            : source === 'autodl_api_discovered'
	                              ? ' · 平台验证'
	                              : ''
                        const ready = image.readyToStart === false ? ' · 未就绪' : ''
                        return (
                          <option key={id} value={id}>
                            {String(image.displayName || id)}{cuda}{sourceLabel}{ready}
                          </option>
                        )
                      })}
                    </select>
                  </label>
                  <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                    单实例 GPU
                    <select
                      value={cloudGpuCount}
                      onChange={(e) => {
                        const nextCount = Math.max(1, Number(e.target.value) || 1)
                        setCloudGpuCount(nextCount)
                        const exactSku = skuForGpuCount(nextCount)
                        if (exactSku?.skuId) setCloudSkuId(String(exactSku.skuId))
                        invalidateCloudPlan()
                      }}
                      className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                    >
                      {gpuCountOptions.map((count) => (
                        <option key={count} value={count} disabled={selectedStockCount > 0 && count * cloudReplicaCount > selectedStockCount}>
                          {count} 张卡
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                    并行副本
                    <select
                      value={cloudReplicaCount}
                      onChange={(e) => {
                        setCloudReplicaCount(Math.max(1, Number(e.target.value) || 1))
                        invalidateCloudPlan()
                      }}
                      className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                    >
                      {replicaCountOptions.map((count) => (
                        <option key={count} value={count} disabled={selectedStockCount > 0 && count * cloudGpuCount > selectedStockCount}>
                          {count} 个
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                {(selectedCloudSku || selectedCloudImage) && (
                  <div className="mt-3 rounded-md border border-bd/50 bg-sf px-3 py-2 text-xs text-tx2 leading-relaxed">
                    <div className="font-semibold text-tx">当前选择</div>
                    <div className="mt-2 grid grid-cols-2 gap-2 max-[760px]:grid-cols-1">
                      <div className="rounded-md border border-bd/40 bg-bg px-2.5 py-2">
                        <div className="text-[11px] font-semibold text-tx3">机型</div>
                        <div className="mt-0.5 text-tx2">
                          {selectedCloudSku
                            ? selectedMachineLabel || String(selectedCloudSku.gpuSpec || selectedCloudSku.skuId)
                            : '平台推荐'}
                        </div>
                        <div className="mt-1 text-[11px] text-tx3">
                          {[
                            selectedHourlyCostCents ? `¥${(selectedHourlyCostCents / 100).toFixed(2)}/小时` : '',
                            selectedCloudSku && selectedSkuGpuCount > 1 && selectedStockCount ? `可开 ${selectedStockCount} 台` : '',
                            selectedAvailableGpuCount ? `空闲 ${selectedAvailableGpuCount} 张卡` : (selectedCloudSku ? '库存启动前确认' : ''),
                            cloudRegionSign
                              ? `地区 ${autodlRegionLabel(cloudRegionSign)}`
                              : selectedRecommendedRegion
                                ? `推荐地区 ${autodlRegionLabel(selectedRecommendedRegion)}`
                                : (selectedCloudSku ? '地区 平台自动择区' : ''),
                            !cloudRegionSign && selectedRegionCount > 1 ? `覆盖 ${selectedRegionCount} 个地区` : '',
                          ].filter(Boolean).join(' · ')}
                        </div>
                      </div>
                      <div className="rounded-md border border-bd/40 bg-bg px-2.5 py-2">
                        <div className="text-[11px] font-semibold text-tx3">本次资源请求</div>
                        <div className="mt-0.5 text-tx2">
                          单实例 {cloudGpuCount} 张卡 × {cloudReplicaCount} 个副本 = {requestedGpuTotal} 张 GPU
                        </div>
                        <div className={`mt-1 text-[11px] ${resourceRequestTooLarge ? 'text-rd' : 'text-tx3'}`}>
                          {resourceRequestTooLarge
                            ? selectedSkuTooSmallForContainer
                              ? `当前选择的是 ${selectedSkuGpuCount} 卡实例，不能承载单实例 ${cloudGpuCount} 卡。`
                              : `超过当前空闲卡 ${selectedAvailableGpuCount} 张，请减少数量或换机型。`
                            : selectedAvailableGpuCount
                              ? `不超过当前看到的空闲卡 ${selectedAvailableGpuCount} 张。`
                              : '库存数量由云平台启动前再次确认。'}
                        </div>
                      </div>
                    </div>
                    <div className="mt-2 text-tx3">
                      {selectedCloudImage
                        ? `镜像：${String(selectedCloudImage.displayName || selectedCloudImage.imageId)}`
                        : '镜像：平台推荐'}
                    </div>
                    {selectedRegionRows.length > 0 && (
                      <div className="mt-3 rounded-md border border-bd/40 bg-bg px-2.5 py-2">
                        <div className="flex items-center justify-between gap-2">
                          <div className="text-[11px] font-semibold text-tx3">分区库存</div>
                          <div className="text-[11px] text-tx3">
                            启动时由 AutoDL 在可用区里最终确认同机容量
                          </div>
                        </div>
                        <div className="mt-2 grid grid-cols-3 gap-2 max-[980px]:grid-cols-2 max-[640px]:grid-cols-1">
                          {selectedRegionRows.map((region, index) => {
                            const instances = cloudGpuCount > 0
                              ? Math.floor(region.availableGpuCount / Math.max(1, cloudGpuCount))
                              : 0
                            const enoughForThisRequest = region.availableGpuCount >= requestedGpuTotal
                            return (
                              <div
                                key={`${region.regionSign || region.label}-${index}`}
                                className={`rounded-md border px-2.5 py-2 ${
                                  enoughForThisRequest
                                    ? 'border-gn/30 bg-gn/5'
                                    : 'border-bd/40 bg-sf'
                                }`}
                              >
                                <div className="flex items-center justify-between gap-2">
                                  <span className="font-semibold text-tx">
                                    {region.label || region.regionSign || '未知区域'}
                                  </span>
                                  <span className="text-[10px] text-tx3">AutoDL</span>
                                </div>
                                <div className="mt-1 text-[11px] text-tx3">
                                  空闲 {region.availableGpuCount} 张卡
                                  {cloudGpuCount > 1 ? ` · 约可开 ${instances} 台 ${cloudGpuCount} 卡实例` : ''}
                                </div>
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                )}
                {cloudResourceCatalog && cloudResourceCatalog.skus.length === 0 && cloudResourceCatalog.images.length === 0 && (
                  <div className="mt-3 rounded-md border border-yl/40 bg-yl/10 px-3 py-2 text-xs text-tx2 leading-relaxed">
                    当前平台还没有返回可选机型和镜像。普通用户不需要填写 token 或实例信息，管理员需要先在后端配置资源 catalog。
                  </div>
                )}
                <div className={`mt-3 rounded-md border px-3 py-2 text-xs leading-relaxed ${
                  runtimeIsIncompatible
                    ? 'border-rd/40 bg-rd/10 text-tx2'
                    : 'border-gn/35 bg-gn/10 text-tx2'
                }`}>
                  <div className="font-semibold text-tx">{runtimeReadyLabel}</div>
                  {runtimeCandidate && (
                    <div className="mt-1 text-tx3">
                      {[
                        runtimeCandidateSkuId ? `机型=${runtimeCandidateSkuId}` : '',
                        runtimeCandidateImageId ? `镜像=${runtimeCandidateImageId}` : '',
                        typeof runtimeCandidate.score === 'number' ? `匹配分=${runtimeCandidate.score}` : '',
                      ].filter(Boolean).join('；')}
                    </div>
                  )}
                  {runtimeBlocking.length > 0 && (
                    <div className="mt-1 text-rd">
                      {runtimeBlockingMessages.slice(0, 2).join('；')}
                    </div>
                  )}
                  {runtimeRisks.length > 0 && (
                    <div className="mt-1 text-yl">
                      {runtimeRisks.slice(0, 2).join('；')}
                    </div>
                  )}
                </div>
              </div>

              <div className="rounded-lg border border-bd/50 bg-bg p-3">
                <div className="mb-3 flex items-start justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold text-tx">调试：覆盖数据来源</div>
                    <div className="mt-1 text-xs text-tx3 leading-relaxed">
                      用户主路径是在上方输入需求或粘贴数据链接。这里仅用于开发调试时强制切换平台数据、公开链接或用户云端数据。
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3 max-[700px]:grid-cols-1">
                  <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                    数据来源
                    <select
                      value={cloudSourceKind}
                      onChange={(e) => {
                        setCloudSourceKind(e.target.value as CloudSourceKind)
                        invalidateCloudPlan()
                      }}
                      className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                    >
                      {CLOUD_SOURCE_OPTIONS.map(option => (
                        <option key={option.id} value={option.id}>{option.title}</option>
                      ))}
                    </select>
                  </label>
                  {cloudSourceKind === 'official_reference' ? (
                    <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                      官方数据
                      <input
                        value={officialDatasetSource?.label || '当前 benchmark 暂无内置数据源'}
                        readOnly
                        className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none"
                      />
                    </label>
                  ) : cloudSourceKind === 'platform_dataset' ? (
                    <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                      平台数据
                      <select
                        value={cloudDatasetId}
                        onChange={(e) => {
                          setCloudDatasetId(e.target.value)
                          invalidateCloudPlan()
                        }}
                        className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                      >
                        <option value="">选择我的 private 数据或已可用 public 数据</option>
                        {trainableDatasets.map(dataset => (
                          <option key={dataset.id} value={dataset.id}>
                            {dataset.label} · {dataset.kind}
                          </option>
                        ))}
                      </select>
                    </label>
                  ) : (
                    <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                      {cloudSourceKind === 'public_reference' ? '数据集链接' : '我的数据集链接'}
                      <input
                        value={cloudSourceUri}
                        onChange={(e) => {
                          setCloudSourceUri(e.target.value)
                          invalidateCloudPlan()
                        }}
                        placeholder={
                          cloudSourceKind === 'public_reference'
                            ? 'HuggingFace / ModelScope 地址，留空由 AI 自动匹配'
                            : '云盘或对象存储地址，例如 s3://...、oss://...、https://...'
                        }
                        className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                      />
                    </label>
                  )}
                </div>
                <div className="mt-3 rounded-md bg-sf2 border border-bd/50 px-3 py-2 text-xs text-tx3 leading-relaxed">
                  {CLOUD_SOURCE_OPTIONS.find(option => option.id === cloudSourceKind)?.detail}
                </div>
                {cloudSourceKind === 'public_reference' && normalizedPublicDatasetUri && (
                  <div className="mt-3 rounded-md border border-bd/50 bg-bg px-3 py-2 text-xs text-tx2 leading-relaxed">
                    <div className="font-semibold text-tx">
                      {sourcePreflightLoading ? '正在预检查公开数据源...' : `公开源预检查 · ${sourcePreflightSize}`}
                    </div>
                    {sourcePreflightPath && (
                      <div className="mt-1 text-tx3">云端缓存：已规划，启动后由远端任务处理。</div>
                    )}
                    {sourcePreflightWarnings.length > 0 && (
                      <div className="mt-1 text-yl">{sourcePreflightWarnings.slice(0, 2).join('；')}</div>
                    )}
                    <div className="mt-1 text-tx3">
                      启动前会让用户确认下载、存储和训练消耗；大小未知也可以继续。
                    </div>
                  </div>
                )}
              </div>

              {canShowInternalTrainingDebug && (
              <div className="rounded-lg border border-bd/50 bg-bg">
                <button
                  type="button"
                  onClick={() => setShowExpertCloudOptions(value => !value)}
                  className="w-full flex items-center justify-between gap-3 px-3 py-2 text-left text-sm font-semibold text-tx hover:bg-sf2 rounded-lg transition-colors"
                >
                  <span>专家参数</span>
                  <span className="text-xs text-tx3">
                    {showExpertCloudOptions ? '收起' : '评测环境 / 格式 / 权限 / RLinf'}
                  </span>
                </button>
                {showExpertCloudOptions && (
                  <div className="border-t border-bd/50 p-3 space-y-3">
                    <div>
                      <div className="text-sm font-semibold text-tx">评测环境 / Benchmark</div>
                      <div className="mt-1 text-xs text-tx3 leading-relaxed">
                        这里是常用快捷预设，不是完整能力清单。其他 benchmark、真机或自定义模拟器可以写进 AI 需求。
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-3 max-[700px]:grid-cols-1">
                      <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                        环境用途
                        <select
                          value={cloudEnvironmentMode}
                          onChange={(e) => setCloudEnvironmentMode(e.target.value as 'none' | 'benchmark')}
                          className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                        >
                          <option value="none">不使用 benchmark，仅用数据训练</option>
                          <option value="benchmark">使用 benchmark 做 RL / 评测</option>
                        </select>
                      </label>
                      {cloudEnvironmentMode === 'benchmark' && (
                        <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                          Benchmark
                          <select
                            value={cloudBenchmark}
                            onChange={(e) => setCloudBenchmark(e.target.value)}
                            className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                          >
                            {BUILTIN_BENCHMARKS.map(item => (
                              <option key={item.id} value={item.id}>{item.label}</option>
                            ))}
                          </select>
                        </label>
                      )}
                    </div>
                    {cloudEnvironmentMode === 'benchmark' && cloudBenchmark === 'custom' && (
                      <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                        自定义环境 / 评测说明
                        <input
                          value={cloudEnvironmentHint}
                          onChange={(e) => setCloudEnvironmentHint(e.target.value)}
                          placeholder="例如 RoboCasa、CALVIN、RLBench、自定义 gym env、真机 SO-101 eval"
                          className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                        />
                      </label>
                    )}
                    <div className="grid grid-cols-2 gap-3 max-[520px]:grid-cols-1">
                      <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                        用户
                        <input
                          value={cloudUsername}
                          onChange={(e) => setCloudUsername(e.target.value)}
                          className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                        />
                      </label>
                      <div className="rounded-md border border-bd/50 bg-sf2 px-3 py-2 text-xs text-tx2 leading-relaxed">
                        <div className="font-semibold text-tx">执行方式自动选择</div>
                        <div className="mt-1 text-tx3">
                          用户只需要描述目标。AI 和云端后端会选择 RLinf、LeRobot、OpenVLA-OFT 或自定义 recipe，并在确认页展示可审查计划。
                        </div>
                      </div>
                      {isRlinfExecution && (
                        <div className="col-span-2 rounded-md border border-ac/30 bg-ac/5 px-3 py-2 text-xs text-tx2 leading-relaxed max-[520px]:col-span-1">
                          <div className="flex items-start justify-between gap-3 max-[760px]:flex-col">
                            <div>
                              <div className="font-semibold text-tx">内部 recipe：RLinf VLA</div>
                              <div className="mt-1 text-tx3">
                                平台会把自然语言需求转换成 RLinf runner/actor/rollout/env contract，并先做 smoke check；普通用户不用手动配置 runner。
                              </div>
                              <div className="mt-2 text-[11px] text-tx3">
                                内部路径：EmbodiedRunner / HybridComponentPlacement / Actor-Rollout-Env workers
                              </div>
                            </div>
                            <div className="grid grid-cols-2 gap-2 min-w-[320px] max-[760px]:w-full max-[760px]:min-w-0">
                              <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                                algorithm
                                <select
                                  value={rlinfAlgorithm}
                                  onChange={(e) => setRlinfAlgorithm(e.target.value)}
                                  className="bg-sf border border-bd text-tx px-2 py-1.5 rounded-md text-xs focus:outline-none focus:border-ac"
                                >
                                  <option value="auto">AI/后端选择</option>
                                  <option value="ppo">PPO</option>
                                  <option value="grpo">GRPO</option>
                                  <option value="sac">SAC</option>
                                </select>
                              </label>
                              <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                                placement
                                <select
                                  value={rlinfPlacementStrategy}
                                  onChange={(e) => setRlinfPlacementStrategy(e.target.value)}
                                  className="bg-sf border border-bd text-tx px-2 py-1.5 rounded-md text-xs focus:outline-none focus:border-ac"
                                >
                                  <option value="single_node">single_node</option>
                                  <option value="hybrid">hybrid</option>
                                </select>
                              </label>
                              <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                                rollout backend
                                <select
                                  value={rlinfRolloutBackend}
                                  onChange={(e) => setRlinfRolloutBackend(e.target.value)}
                                  className="bg-sf border border-bd text-tx px-2 py-1.5 rounded-md text-xs focus:outline-none focus:border-ac"
                                >
                                  <option value="huggingface">huggingface</option>
                                  <option value="sglang">sglang</option>
                                  <option value="vllm">vllm</option>
                                </select>
                              </label>
                              <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                                group size
                                <input
                                  type="number"
                                  min={1}
                                  max={64}
                                  value={rlinfGroupSize}
                                  onChange={(e) => setRlinfGroupSize(Math.max(1, Number(e.target.value) || 1))}
                                  className="bg-sf border border-bd text-tx px-2 py-1.5 rounded-md text-xs font-mono focus:outline-none focus:border-ac"
                                />
                              </label>
                            </div>
                          </div>
                        </div>
                      )}
                      <div className="col-span-2 rounded-md border border-bd/50 bg-sf2 px-3 py-2 text-xs text-tx2 leading-relaxed max-[520px]:col-span-1">
                        <div className="flex items-start justify-between gap-3 max-[640px]:flex-col">
                          <div>
                            <div className="font-semibold text-tx">数据和模型格式自动识别</div>
                            <div className="mt-1 text-tx3">
                              通常不需要手动选择。当前识别：数据 {labeledFormat(inferredDatasetFormat)}，权重 {labeledFormat(inferredCheckpointFormat)}。
                            </div>
                          </div>
                          <button
                            type="button"
                            onClick={() => setShowFormatOverrides(value => !value)}
                            className="shrink-0 px-2.5 py-1 rounded-md border border-bd text-[11px] font-semibold text-tx3 hover:text-ac hover:border-ac transition-colors"
                          >
                            {showFormatOverrides ? '收起格式选择' : '手动选择格式'}
                          </button>
                        </div>
                        {showFormatOverrides && (
                          <div className="mt-3 grid grid-cols-2 gap-3 max-[640px]:grid-cols-1">
                            <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                              数据格式
                              <select
                                value={cloudFormat}
                                onChange={(e) => setCloudFormat(e.target.value)}
                                className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                              >
                                {DATASET_FORMATS.map(format => (
                                  <option key={format} value={format}>{labeledFormat(format)}</option>
                                ))}
                              </select>
                            </label>
                            <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                              模型权重格式
                              <select
                                value={cloudCheckpointFormat}
                                onChange={(e) => setCloudCheckpointFormat(e.target.value)}
                                className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                              >
                                {CHECKPOINT_FORMATS.map(format => (
                                  <option key={format} value={format}>{labeledFormat(format)}</option>
                                ))}
                              </select>
                            </label>
                            <div className="col-span-2 text-[11px] text-tx3 leading-relaxed max-[640px]:col-span-1">
                              这里只给开发/兼容场景使用。普通用户贴数据链接、选择模型或上传 checkpoint 后，AI 和云端 preflight 应该自动完成识别。
                            </div>
                          </div>
                        )}
                      </div>
                      <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                        数据访问权限
                        <select
                          value={cloudDataAccessMode}
                          onChange={(e) => setCloudDataAccessMode(e.target.value as AccessMode)}
                          className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                        >
                          <option value="public">公开 / 已签名链接，无需平台授权</option>
                          <option value="saved_connection">使用后端已保存授权连接</option>
                        </select>
                      </label>
                      {cloudDataAccessMode === 'saved_connection' && (
                        <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                          数据授权连接 ID
                          <select
                            value={cloudAuthRef}
                            onChange={(e) => setCloudAuthRef(e.target.value)}
                            className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                            disabled={dataAuthConnections.length === 0}
                          >
                            <option value="">
                              {dataAuthConnections.length ? '选择后端已配置的数据连接' : '后端还没有配置数据连接'}
                            </option>
                            {dataAuthConnections.map(connection => (
                              <option key={connection.id} value={connection.id} disabled={!connection.configured}>
                                {connection.label || connection.id} · {connection.provider}{connection.configured ? '' : ' · 密钥未完整配置'}
                              </option>
                            ))}
                          </select>
                          {dataAuthConnections.length === 0 && (
                            <span className="text-[11px] text-tx3 normal-case">
                              需要管理员在服务器配置授权连接表；页面不会收集 token 或密钥。
                            </span>
                          )}
                        </label>
                      )}
                      <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                        模型访问权限
                        <select
                          value={cloudModelAccessMode}
                          onChange={(e) => setCloudModelAccessMode(e.target.value as AccessMode)}
                          className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                        >
                          <option value="public">公开来源 / 无需授权</option>
                          <option value="saved_connection">使用后端已保存授权连接</option>
                        </select>
                      </label>
                      {cloudModelAccessMode === 'saved_connection' && (
                        <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                          模型授权连接 ID
                          <select
                            value={cloudModelAuthRef}
                            onChange={(e) => setCloudModelAuthRef(e.target.value)}
                            className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                            disabled={modelAuthConnections.length === 0}
                          >
                            <option value="">
                              {modelAuthConnections.length ? '选择后端已配置的模型连接' : '后端还没有配置模型连接'}
                            </option>
                            {modelAuthConnections.map(connection => (
                              <option key={connection.id} value={connection.id} disabled={!connection.configured}>
                                {connection.label || connection.id} · {connection.provider}{connection.configured ? '' : ' · 密钥未完整配置'}
                              </option>
                            ))}
                          </select>
                          {modelAuthConnections.length === 0 && (
                            <span className="text-[11px] text-tx3 normal-case">
                              需要管理员在服务器配置授权连接表；页面不会收集 token 或密钥。
                            </span>
                          )}
                        </label>
                      )}
                    </div>
                    <div className="rounded-md bg-sf2 border border-bd/50 px-3 py-2 text-xs text-tx3 leading-relaxed">
                      这里不是填写 public/private，也不是填写链接。公开数据和公开模型选择“无需授权”；私有 OSS/S3/R2/企业仓库应先在后端保存凭证，页面只会选择后端真的存在的授权连接 ID。
                    </div>
                    <div className="rounded-md border border-bd/70 bg-bg/60 p-3">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <div className="text-xs font-semibold text-tx">私有仓库 / 私有云存储连接</div>
                          <div className="text-[11px] text-tx3 mt-1">
                            私有 HuggingFace、ModelScope、Git 或对象存储需要先保存连接；token 只发给后端一次，不进训练参数。
                          </div>
                        </div>
                        <button
                          type="button"
                          onClick={() => setShowAuthConnectionForm(!showAuthConnectionForm)}
                          className="px-3 py-1.5 rounded-md border border-bd text-xs text-tx hover:border-ac"
                        >
                          {showAuthConnectionForm ? '收起' : '添加连接'}
                        </button>
                      </div>
                      {showAuthConnectionForm && (
                        <div className="grid grid-cols-2 gap-3 mt-3 max-[760px]:grid-cols-1">
                          <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                            用途
                            <select
                              value={authConnectionKind}
                              onChange={(e) => setAuthConnectionKind(e.target.value as 'data' | 'model')}
                              className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                            >
                              <option value="data">私有数据集</option>
                              <option value="model">私有模型 / checkpoint</option>
                            </select>
                          </label>
                          <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                            来源类型
                            <select
                              value={authConnectionProvider}
                              onChange={(e) => setAuthConnectionProvider(e.target.value)}
                              className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                            >
                              <option value="huggingface">HuggingFace 私有仓库</option>
                              <option value="modelscope">ModelScope 私有仓库</option>
                              <option value="git">Git 私有仓库</option>
                              <option value="s3">S3 兼容存储</option>
                              <option value="oss">阿里云 OSS</option>
                              <option value="r2">Cloudflare R2</option>
                              <option value="cos">腾讯云 COS</option>
                              <option value="minio">MinIO</option>
                              <option value="kaggle">Kaggle</option>
                              <option value="dagshub">DagsHub</option>
                              <option value="custom">自定义</option>
                            </select>
                          </label>
                          <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                            连接 ID
                            <input
                              value={authConnectionId}
                              onChange={(e) => setAuthConnectionId(e.target.value)}
                              placeholder="例如 hf-private-models"
                              className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                            />
                          </label>
                          <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                            显示名称
                            <input
                              value={authConnectionLabel}
                              onChange={(e) => setAuthConnectionLabel(e.target.value)}
                              placeholder="例如 我的 HF 私有模型"
                              className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                            />
                          </label>
                          {usesObjectStorageSecret ? (
                            <>
                              <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                                Access Key ID
                                <input
                                  value={authConnectionAccessKey}
                                  onChange={(e) => setAuthConnectionAccessKey(e.target.value)}
                                  placeholder="只保存到后端"
                                  className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                                />
                              </label>
                              <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                                Secret Access Key
                                <input
                                  type="password"
                                  value={authConnectionSecretKey}
                                  onChange={(e) => setAuthConnectionSecretKey(e.target.value)}
                                  placeholder="不会显示在列表和训练参数里"
                                  className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                                />
                              </label>
                            </>
                          ) : (
                            <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono col-span-2 max-[760px]:col-span-1">
                              Token / 访问令牌
                              <input
                                type="password"
                                value={authConnectionToken}
                                onChange={(e) => setAuthConnectionToken(e.target.value)}
                                placeholder="例如 HuggingFace read token；不会显示在列表和训练参数里"
                                className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                              />
                            </label>
                          )}
                          <div className="col-span-2 flex items-center gap-3 max-[760px]:col-span-1">
                            <button
                              type="button"
                              onClick={() => { void savePrivateAuthConnection() }}
                              className="px-3 py-2 rounded-md bg-ac text-white text-xs font-semibold hover:bg-ac/90"
                            >
                              保存到后端连接库
                            </button>
                            {authConnectionMessage && (
                              <span className="text-xs text-tx3">{authConnectionMessage}</span>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
              )}
                </>
              )}
            </div>
          )}

          {trainMode === 'local' && (
            <>
          <div className="flex gap-3 mb-3 max-[700px]:flex-col">
              <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono flex-1">
                {t('policyType')}
                <select
                  value={policyType}
                  onChange={(e) => setPolicyType(e.target.value)}
                  className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                >
                  {POLICY_TYPES.map(type => (
                    <option key={type} value={type}>{type}</option>
                  ))}
                </select>
              </label>
            <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono flex-1">
              {t('steps')}
              <input type="number" value={trainSteps} onChange={(e) => setTrainSteps(Number(e.target.value) || 100000)}
                className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac" />
            </label>
            <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono w-[90px]">
              {t('device')}
              <select value={trainDevice} onChange={(e) => setTrainDevice(e.target.value)}
                className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac">
                <option value="cuda">cuda</option>
                <option value="cpu">cpu</option>
              </select>
            </label>
          </div>
          <div className="flex gap-3 max-[520px]:flex-col">
            <button
              disabled={!canStartTraining || !trainDataset}
              title={startBlockers[0] || ''}
              onClick={() => {
                void doTrainStart({
                  dataset_name: trainDataset,
                  policy_type: policyType,
                  steps: trainSteps,
                  device: trainDevice,
                })
              }}
              className="flex-1 px-4 py-2.5 rounded-lg text-sm font-semibold text-white bg-ac hover:bg-ac2 shadow-glow-ac
                transition-all active:scale-[0.97] disabled:opacity-25 disabled:cursor-not-allowed disabled:shadow-none"
            >
              {trainingLoading ? t('startingTraining') : t('startTraining')}
            </button>
            <button
              disabled={!currentTrainJobId || !!trainingStopLoading}
              onClick={() => { void doTrainStop() }}
              className="px-4 py-2.5 rounded-lg text-sm font-semibold text-white bg-rd hover:bg-rd/90
                transition-all active:scale-[0.97] disabled:opacity-25 disabled:cursor-not-allowed"
            >
              {trainingStopLoading ? t('stoppingTraining') : t('stopTraining')}
            </button>
          </div>
            </>
          )}
          {trainMode === 'local' && startBlockers.length > 0 && hasCloudPlan && (
            <div className="mt-3 rounded-lg border border-yl/40 bg-yl/10 px-3 py-2 text-xs text-tx2 leading-relaxed">
              <div className="font-semibold text-tx">启动前还需要：</div>
              <ul className="mt-1 list-disc pl-4 space-y-1">
                {startBlockers.map(item => <li key={item}>{item}</li>)}
              </ul>
            </div>
          )}
        </section>

        {trainMode === 'local' && (
        <section className="bg-sf rounded-xl p-5 shadow-card shadow-inset-gn">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-bold text-tx uppercase tracking-wide">{t('policies') || 'Policies'}</h3>
            <button
              onClick={() => { void loadPolicies() }}
              className="px-2.5 py-0.5 bg-ac/10 text-ac rounded text-xs font-medium hover:bg-ac/20 transition-colors"
            >
              {t('refresh')}
            </button>
          </div>

          {policies.length === 0 && (
            <div className="text-tx3 text-center py-4 text-sm">{t('noPolicies')}</div>
          )}
          <div className="space-y-1.5">
            {policies.map((p: any, i: number) => (
              <div key={i} className="bg-bg border border-bd/30 rounded-lg px-3 py-2 text-sm flex items-center gap-2">
                <span className="flex-1 font-mono text-tx2 truncate">
                  {typeof p === 'string' ? p : p.name || JSON.stringify(p)}
                </span>
                <button
                  disabled={!!hubLoading}
                  onClick={() => promptPushPolicy(typeof p === 'string' ? p : p.name)}
                  className="px-2 py-0.5 text-ac/60 rounded text-xs hover:text-ac hover:bg-ac/10 transition-colors disabled:opacity-25"
                >
                  {t('pushToHub')}
                </button>
              </div>
            ))}
          </div>

          <div className="mt-4 pt-4 border-t border-bd/40">
            <h4 className="text-xs font-bold text-tx3 uppercase mb-2">{t('downloadPolicy')}</h4>
            <div className="flex gap-2">
              <input
                placeholder={t('repoIdPlaceholder')}
                value={pullPolicyRepo}
                onChange={(e) => setPullPolicyRepo(e.target.value)}
                className="flex-1 bg-bg border border-bd text-tx px-3 py-1.5 rounded-lg text-sm
                  focus:outline-none focus:border-ac"
              />
              <button
                disabled={!pullPolicyRepo || !!hubLoading}
                onClick={() => {
                  void pullPolicy(pullPolicyRepo)
                  setPullPolicyRepo('')
                }}
                className="px-3 py-1.5 bg-ac/10 text-ac rounded-lg text-sm font-medium
                  hover:bg-ac/20 transition-colors disabled:opacity-25 disabled:cursor-not-allowed"
              >
                {hubLoading === 'pullPolicy' ? t('downloading') : t('download')}
              </button>
            </div>
          </div>

          {hubProgress && !hubProgress.done && hubLoading === 'pullPolicy' && (
            <div className="mt-3">
              <div className="flex items-center justify-between text-2xs text-tx3 mb-1">
                <span>{hubProgress.operation}</span>
                <span>{hubProgress.progress_percent.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-bd/30 rounded-full h-1.5">
                <div
                  className="bg-gn h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(hubProgress.progress_percent, 100)}%` }}
                />
              </div>
            </div>
          )}
        </section>
        )}

        {trainMode === 'local' && <LossCurvePanel />}
        {(trainMode === 'local' || showCloudProgress) && <TrainingProgressPanel />}
      </div>
    </div>
  )
}
