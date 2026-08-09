import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  Aperture,
  Camera,
  Check,
  ChevronRight,
  CircleAlert,
  Cpu,
  Download,
  Eye,
  EyeOff,
  Gauge,
  HardDrive,
  Play,
  RefreshCw,
  RotateCcw,
  ScanFace,
  Search,
  Settings2,
  SlidersHorizontal,
  Sparkles,
  Square,
  Trash2,
  Upload,
  UserRoundPlus,
  Video,
  VideoOff,
  Zap,
} from 'lucide-react'
import { api } from './api'
import type {
  CameraInfo,
  Config,
  DeepModelInfo,
  DfmTrainingJob,
  DfmTrainingStatus,
  IdentityProfile,
  IdentityPrompt,
  IdentitySession,
  ModelInstallStatus,
  ModuleProviderTiming,
  Options,
  ProviderTimingComparison,
  RuntimeStatus,
  TimingsPayload,
} from './types'

type TabId = 'live' | 'trainer' | 'camera' | 'processing' | 'routing' | 'system'

const EMPTY_STATUS: RuntimeStatus = {
  running: false,
  capture_active: false,
  status: 'Connecting…',
  preview_sequence: 0,
  has_preview: false,
  uptime_seconds: 0,
}

const NAV_ITEMS: Array<{ id: TabId; label: string; icon: typeof Video }> = [
  { id: 'live', label: 'Face & output', icon: ScanFace },
  { id: 'trainer', label: 'Identity trainer', icon: UserRoundPlus },
  { id: 'camera', label: 'Camera', icon: Camera },
  { id: 'processing', label: 'Processing', icon: Sparkles },
  { id: 'routing', label: 'Execution routing', icon: Cpu },
  { id: 'system', label: 'System', icon: Settings2 },
]

const CAPTURE_RESTART_KEYS = new Set([
  'backend',
  'camera_choice',
  'resolution_preset',
  'width',
  'height',
  'fps',
  'dshow_name_device',
  'convert_rgb',
  'force_fourcc',
  'retry_black',
  'gentle_mode',
  'auto_repair',
  'color_mode',
  'lock_exposure',
  'exposure_value',
  'lock_wb',
  'wb_temperature',
])

const MODEL_RELOAD_KEYS = new Set(['deep_swapper_model', 'face_swapper_model'])

function formatMetric(value: number | undefined, suffix = ''): string {
  return value === undefined ? '—' : `${value.toFixed(value >= 100 ? 0 : 1)}${suffix}`
}

function timingMean(timing?: ModuleProviderTiming | null): number | undefined {
  return timing?.per_frame?.mean_ms ?? timing?.per_call?.mean_ms
}

function formatTiming(value: number | null | undefined): string {
  if (value === null || value === undefined) return 'not measured'
  if (value < 1) return `${value.toFixed(2)} ms`
  return `${value.toFixed(value >= 100 ? 0 : 1)} ms`
}

function parseResolution(value: string): [number, number] {
  const [width, height] = value.split('x').map(Number)
  return [width || 1920, height || 1080]
}

function SelectField({
  label,
  value,
  options,
  onChange,
  hint,
  disabled,
}: {
  label: string
  value: string | number | undefined
  options: Array<string | number>
  onChange: (value: string) => void
  hint?: string
  disabled?: boolean
}) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <select value={value ?? ''} onChange={(event) => onChange(event.target.value)} disabled={disabled}>
        {options.map((option) => <option value={option} key={String(option)}>{String(option)}</option>)}
      </select>
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  )
}

function RoutingField({
  label,
  value,
  providers,
  comparison,
  liveTiming,
  onChange,
}: {
  label: string
  value: string
  providers: string[]
  comparison?: ProviderTimingComparison
  liveTiming?: Record<string, ModuleProviderTiming>
  onChange: (value: string) => void
}) {
  const cudaTiming = comparison?.cuda_mean_ms ?? timingMean(liveTiming?.cuda)
  const cpuTiming = comparison?.cpu_mean_ms ?? timingMean(liveTiming?.cpu)
  const ratio = comparison?.cpu_to_cuda_ratio ?? (cudaTiming && cpuTiming ? cpuTiming / cudaTiming : undefined)
  const active = comparison ? comparison.active : Boolean(cudaTiming || cpuTiming)
  const selectedTiming = timingMean(liveTiming?.[value])
  const tensorrtPrecision = label === 'Masker' ? 'FP16' : 'FP32'
  const tensorrtLabel = label === 'Masker' ? 'tensorrt / FP16 experimental' : `tensorrt / ${tensorrtPrecision}`

  return (
    <label className={`route-card ${active ? '' : 'unmeasured'}`}>
      <span className="route-card-heading">
        <span className="field-label">{label}</span>
        {ratio !== undefined && <strong>{ratio >= 1 ? `CPU ${ratio.toFixed(1)}x slower` : `CPU ${(1 / ratio).toFixed(1)}x faster`}</strong>}
      </span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {providers.map((option) => <option value={option} key={option}>{option === 'tensorrt' ? tensorrtLabel : option}</option>)}
      </select>
      <span className="route-timings">
        <span className="timing-chip cuda"><small>CUDA</small><strong>{formatTiming(cudaTiming)}</strong></span>
        <span className="timing-chip cpu"><small>CPU</small><strong>{formatTiming(cpuTiming)}</strong></span>
      </span>
      {selectedTiming !== undefined
        ? <span className="route-live"><span /> Live on {value}{value === 'tensorrt' ? ` / ${tensorrtPrecision}` : ''}: {formatTiming(selectedTiming)} / processed frame</span>
        : !active && <span className="route-live muted">Not used by the measured pipeline</span>}
    </label>
  )
}

function formatFileSize(bytes?: number | null): string {
  if (!bytes) return 'Size unknown'
  return bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(1)} GB` : `${Math.round(bytes / 1024 ** 2)} MB`
}

function modelInitials(name: string): string {
  return name
    .replace(/\([^)]*\)/g, '')
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .map((part) => part[0] || '')
    .join('')
    .toUpperCase() || '?'
}

function ModelBrowser({
  models,
  value,
  install,
  disabled,
  onSelect,
  onInstall,
}: {
  models: DeepModelInfo[]
  value: string
  install: ModelInstallStatus | null
  disabled: boolean
  onSelect: (modelId: string) => void
  onInstall: (modelId: string) => void
}) {
  const currentGender = models.find((model) => model.id === value)?.gender || 'male'
  const [gender, setGender] = useState<'all' | 'male' | 'female' | 'other'>(currentGender)
  const [query, setQuery] = useState('')
  const normalizedQuery = query.trim().toLowerCase()
  const counts = {
    all: models.length,
    male: models.filter((model) => model.gender === 'male').length,
    female: models.filter((model) => model.gender === 'female').length,
    other: models.filter((model) => model.gender === 'other').length,
  }
  const hubModelCount = models.filter((model) => model.downloadable).length
  const filteredModels = models
    .filter((model) => normalizedQuery || gender === 'all' || model.gender === gender)
    .filter((model) => !normalizedQuery || `${model.name} ${model.id} ${model.creator}`.toLowerCase().includes(normalizedQuery))
    .sort((left, right) => Number(right.installed) - Number(left.installed) || left.name.localeCompare(right.name))

  return (
    <div className="model-browser">
      <label className="model-search">
        <Search size={16} />
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search people, creators or resolution…" aria-label="Search face models" />
        {query && <button type="button" onClick={() => setQuery('')}>Clear</button>}
      </label>
      <div className="model-gender-tabs" role="tablist" aria-label="Model gender">
        {(['all', 'male', 'female'] as const).map((item) => (
          <button type="button" role="tab" aria-selected={gender === item} className={gender === item ? 'active' : ''} onClick={() => setGender(item)} key={item}>
            {item === 'all' ? 'All' : item === 'male' ? 'Male' : 'Female'} <span>{counts[item]}</span>
          </button>
        ))}
        {counts.other > 0 && <button type="button" role="tab" aria-selected={gender === 'other'} className={gender === 'other' ? 'active' : ''} onClick={() => setGender('other')}>Other <span>{counts.other}</span></button>}
      </div>
      <div className="model-browser-summary">
        <span>{normalizedQuery ? `${filteredModels.length} search result${filteredModels.length === 1 ? '' : 's'}` : `${filteredModels.length} ${gender === 'all' ? 'models' : `${gender} models`}`}</span>
        <span><i className="installed-dot" /> Installed <i className="hub-dot" /> Available on Hub</span>
      </div>
      {hubModelCount > 0 && !normalizedQuery && (
        <div className="model-hub-callout">
          <Download size={16} />
          <span><strong>{hubModelCount} additional compatible models found</strong><small>Install individually from the Hugging Face community library.</small></span>
          <button type="button" onClick={() => { setGender('all'); setQuery('Hugging Face community') }}>Show</button>
        </div>
      )}
      <div className="model-card-grid">
        {filteredModels.map((model) => {
          const selected = model.id === value
          const downloading = Boolean(install?.running && install.model_id === model.id)
          const anotherDownload = Boolean(install?.running && install.model_id !== model.id)
          return (
            <button
              type="button"
              className={`model-card ${selected ? 'selected' : ''} ${model.installed ? 'installed' : 'hub-model'}`}
              onClick={() => model.installed ? onSelect(model.id) : onInstall(model.id)}
              disabled={disabled || anotherDownload}
              title={model.installed ? `Select ${model.name}` : `Download ${model.name} from Hugging Face (${formatFileSize(model.size_bytes)})`}
              key={model.id}
            >
              <span className="model-thumb-wrap">
                <span className="model-thumb-fallback" aria-hidden="true">{modelInitials(model.name)}</span>
                <img
                  src={model.thumbnail_url}
                  alt={`${model.name} model`}
                  loading="lazy"
                  onError={(event) => { event.currentTarget.style.display = 'none' }}
                />
                {selected && <span className="model-selected-check"><Check size={13} /></span>}
                {!model.installed && <span className="model-hub-badge"><Download size={11} /> Hub</span>}
                {downloading && <span className="model-download-progress"><span style={{ width: `${Math.max(2, (install?.progress || 0) * 100)}%` }} /></span>}
              </span>
              <span className="model-card-copy">
                <strong>{model.name}</strong>
                <small>{model.resolution ? `${model.resolution}px` : 'DFM'} · {model.creator}</small>
                {!model.installed && <em>{downloading ? `Downloading ${Math.round((install?.progress || 0) * 100)}%` : `${formatFileSize(model.size_bytes)} · Install`}</em>}
              </span>
            </button>
          )
        })}
      </div>
      {filteredModels.length === 0 && <div className="model-empty"><Search size={20} /><strong>No matching models</strong><span>Try a different name or clear the search.</span></div>}
      {install?.error && <div className="timing-error"><CircleAlert size={15} />{install.error}</div>}
    </div>
  )
}

function RangeField({
  label,
  value,
  min,
  max,
  step = 1,
  unit,
  onChange,
  onCommit,
}: {
  label: string
  value: number
  min: number
  max: number
  step?: number
  unit?: string
  onChange: (value: number) => void
  onCommit?: () => void
}) {
  return (
    <label className="field range-field">
      <span className="field-label range-label"><span>{label}</span><strong>{value}{unit}</strong></span>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
        onPointerUp={onCommit}
        onKeyUp={onCommit}
      />
      <span className="range-ends"><span>{min}</span><span>{max}</span></span>
    </label>
  )
}

function Toggle({
  label,
  description,
  checked,
  onChange,
  accent = false,
}: {
  label: string
  description?: string
  checked: boolean
  onChange: (checked: boolean) => void
  accent?: boolean
}) {
  return (
    <label className={`toggle-row ${accent ? 'accent-toggle' : ''}`}>
      <span>
        <strong>{label}</strong>
        {description && <small>{description}</small>}
      </span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span className="switch" aria-hidden="true"><span /></span>
    </label>
  )
}

function Panel({ title, description, children, className = '' }: { title: string; description?: string; children: React.ReactNode; className?: string }) {
  return (
    <section className={`control-panel ${className}`}>
      <div className="panel-heading">
        <div>
          <h3>{title}</h3>
          {description && <p>{description}</p>}
        </div>
      </div>
      <div className="panel-body">{children}</div>
    </section>
  )
}

function App() {
  const [config, setConfig] = useState<Config | null>(null)
  const [options, setOptions] = useState<Options | null>(null)
  const [camera, setCamera] = useState<CameraInfo | null>(null)
  const [runtime, setRuntime] = useState<RuntimeStatus>(EMPTY_STATUS)
  const [activeTab, setActiveTab] = useState<TabId>('live')
  const [dirtyKeys, setDirtyKeys] = useState<Set<string>>(new Set())
  const [busy, setBusy] = useState(false)
  const [cameraBusy, setCameraBusy] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const [previewVisible, setPreviewVisible] = useState(() => window.localStorage.getItem('faceflow-preview-hidden') !== 'true')
  const [timings, setTimings] = useState<TimingsPayload | null>(null)
  const [modelCatalog, setModelCatalog] = useState<DeepModelInfo[]>([])
  const [modelInstall, setModelInstall] = useState<ModelInstallStatus | null>(null)
  const [identityProfiles, setIdentityProfiles] = useState<IdentityProfile[]>([])
  const [identityPrompts, setIdentityPrompts] = useState<IdentityPrompt[]>([])
  const [identitySession, setIdentitySession] = useState<IdentitySession | null>(null)
  const [identityName, setIdentityName] = useState('')
  const [identityConsent, setIdentityConsent] = useState(false)
  const [identityBusyPrompt, setIdentityBusyPrompt] = useState<string | null>(null)
  const [identityMode, setIdentityMode] = useState<'instant' | 'dfm'>('instant')
  const [dfmStatus, setDfmStatus] = useState<DfmTrainingStatus | null>(null)
  const [dfmSelectedProfile, setDfmSelectedProfile] = useState('')
  const [dfmEngineRoot, setDfmEngineRoot] = useState('')
  const [dfmBaseWorkspace, setDfmBaseWorkspace] = useState('')
  const [dfmTargetIterations, setDfmTargetIterations] = useState(1_000_000)
  const [dfmBusyAction, setDfmBusyAction] = useState<string | null>(null)
  const [fatalError, setFatalError] = useState<string | null>(null)
  const dirtyCountRef = useRef(0)
  const pendingInstallModel = useRef<string | null>(null)

  const showNotice = useCallback((message: string) => {
    setNotice(message)
    window.setTimeout(() => setNotice(null), 3400)
  }, [])

  useEffect(() => {
    api.bootstrap()
      .then((data) => {
        setConfig(data.config)
        setOptions(data.options)
        setModelCatalog(data.options.deep_model_catalog || [])
        setCamera(data.camera)
        setRuntime(data.runtime)
      })
      .catch((error) => setFatalError(error.message))
    api.timings().then(setTimings).catch(() => undefined)
    api.models().then((data) => {
      setModelCatalog(data.models)
      setModelInstall(data.install)
    }).catch(() => undefined)
    api.identities().then((data) => {
      setIdentityProfiles(data.profiles)
      setIdentityPrompts(data.prompts)
    }).catch(() => undefined)
    api.dfmStatus().then(setDfmStatus).catch(() => undefined)
  }, [])

  useEffect(() => {
    if (!dfmEngineRoot && dfmStatus?.environment.engine_root) setDfmEngineRoot(dfmStatus.environment.engine_root)
  }, [dfmEngineRoot, dfmStatus?.environment.engine_root])

  useEffect(() => {
    if (!dfmSelectedProfile && identityProfiles.length) {
      setDfmSelectedProfile((identityProfiles.find((profile) => profile.dfm_ready) || identityProfiles[0]).id)
    }
  }, [dfmSelectedProfile, identityProfiles])

  useEffect(() => {
    if (activeTab !== 'trainer') return
    const hasActiveJob = dfmStatus?.jobs.some((job) => ['preparing', 'extracting', 'training', 'stopping', 'exporting'].includes(job.phase))
    if (identityMode !== 'dfm' && !hasActiveJob) return
    const timer = window.setInterval(() => api.dfmStatus().then(setDfmStatus).catch(() => undefined), hasActiveJob ? 1200 : 5000)
    return () => window.clearInterval(timer)
  }, [activeTab, dfmStatus?.jobs, identityMode])

  useEffect(() => {
    dirtyCountRef.current = dirtyKeys.size
  }, [dirtyKeys])

  useEffect(() => {
    window.localStorage.setItem('faceflow-preview-hidden', String(!previewVisible))
  }, [previewVisible])

  useEffect(() => {
    const timer = window.setInterval(() => {
      if (dirtyCountRef.current === 0) {
        api.config().then(setConfig).catch(() => undefined)
      }
    }, 3000)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    const timer = window.setInterval(() => {
      api.timings().then(setTimings).catch(() => undefined)
    }, timings?.benchmark.running ? 500 : 1000)
    return () => window.clearInterval(timer)
  }, [timings?.benchmark.running])

  useEffect(() => {
    if (!modelInstall?.running) return
    const timer = window.setInterval(() => {
      api.models().then((data) => {
        setModelCatalog(data.models)
        setModelInstall(data.install)
      }).catch(() => undefined)
    }, 650)
    return () => window.clearInterval(timer)
  }, [modelInstall?.running])

  useEffect(() => {
    const timer = window.setInterval(() => {
      api.status().then((next) => {
        setRuntime(next)
      }).catch(() => undefined)
    }, 650)
    return () => window.clearInterval(timer)
  }, [])

  const stage = useCallback((key: string, value: unknown) => {
    setConfig((current) => current ? { ...current, [key]: value } : current)
    setDirtyKeys((current) => new Set(current).add(key))
  }, [])

  useEffect(() => {
    if (modelInstall?.error && pendingInstallModel.current === modelInstall.model_id) {
      showNotice(modelInstall.error)
      pendingInstallModel.current = null
    } else if (modelInstall?.complete && pendingInstallModel.current === modelInstall.model_id && modelInstall.model_id) {
      stage('deep_swapper_model', modelInstall.model_id)
      showNotice('Model installed and selected; apply when you are ready')
      pendingInstallModel.current = null
    }
  }, [modelInstall, showNotice, stage])

  const installDeepModel = useCallback(async (modelId: string) => {
    pendingInstallModel.current = modelId
    try {
      const status = await api.installModel(modelId)
      setModelInstall(status)
      if (status.complete) {
        const catalog = await api.models()
        setModelCatalog(catalog.models)
        setModelInstall(catalog.install)
      } else {
        showNotice('Downloading the DFM model from Hugging Face')
      }
    } catch (error) {
      pendingInstallModel.current = null
      showNotice(error instanceof Error ? error.message : 'Model download could not start')
    }
  }, [showNotice])

  const commit = useCallback(async (values: Partial<Config>, restart = false, clearModels = false) => {
    setBusy(true)
    try {
      const result = await api.patchConfig(values, restart && runtime.running, clearModels)
      setConfig(result.config)
      setRuntime(result.runtime)
      setDirtyKeys((current) => {
        const next = new Set(current)
        Object.keys(values).forEach((key) => next.delete(key))
        return next
      })
      return true
    } catch (error) {
      showNotice(error instanceof Error ? error.message : 'Could not save settings')
      return false
    } finally {
      setBusy(false)
    }
  }, [runtime.running, showNotice])

  const applyPending = useCallback(async () => {
    if (!config || dirtyKeys.size === 0) return
    const changedKeys = [...dirtyKeys]
    const values = Object.fromEntries(changedKeys.map((key) => [key, config[key]]))
    const restartRequired = runtime.running && changedKeys.some((key) => CAPTURE_RESTART_KEYS.has(key))
    const clearModels = changedKeys.some((key) => MODEL_RELOAD_KEYS.has(key))
    const saved = await commit(values, restartRequired, clearModels)
    if (saved) {
      showNotice(restartRequired ? 'Settings applied and camera restarted' : runtime.running ? 'Settings applied live' : 'Settings saved')
    }
  }, [commit, config, dirtyKeys, runtime.running, showNotice])

  const selectDeepModel = useCallback(async (modelId: string) => {
    if (!config || modelId === config.deep_swapper_model) return
    if (runtime.running) {
      showNotice('Preparing the new face model; the current face stays live until it is ready')
    }
    const saved = await commit({ deep_swapper_model: modelId }, false, true)
    if (saved) {
      showNotice(runtime.running ? 'New face model warmed and switched between frames' : 'Face model selected')
    }
  }, [commit, config, runtime.running, showNotice])

  const handleRuntime = useCallback(async (action: 'start' | 'stop' | 'restart') => {
    setBusy(true)
    try {
      if (action === 'start' && config && dirtyKeys.size) {
        const values = Object.fromEntries([...dirtyKeys].map((key) => [key, config[key]]))
        const result = await api.patchConfig(values)
        setConfig(result.config)
        setDirtyKeys(new Set())
      }
      const next = action === 'start' ? await api.startCapture() : action === 'stop' ? await api.stopCapture() : await api.restart()
      setRuntime(next)
    } catch (error) {
      showNotice(error instanceof Error ? error.message : `Could not ${action} camera`)
    } finally {
      setBusy(false)
    }
  }, [config, dirtyKeys, showNotice])

  const changeCamera = useCallback(async (choice: string) => {
    setCameraBusy(true)
    try {
      const nextCamera = await api.cameras(true, choice)
      setCamera(nextCamera)
      const resolution = nextCamera.modes[String(config?.resolution_preset)]
        ? String(config?.resolution_preset)
        : nextCamera.resolutions[nextCamera.resolutions.length - 1] || '1920x1080'
      const [width, height] = parseResolution(resolution)
      const maxFps = nextCamera.modes[resolution] || 30
      await commit({
        camera_choice: choice,
        resolution_preset: resolution,
        width,
        height,
        fps: Math.min(Number(config?.fps || 30), maxFps),
      }, true)
    } catch (error) {
      showNotice(error instanceof Error ? error.message : 'Could not switch camera')
    } finally {
      setCameraBusy(false)
    }
  }, [commit, config, showNotice])

  const selectResolution = useCallback((resolution: string) => {
    if (!camera) return
    const [width, height] = parseResolution(resolution)
    const maxFps = camera.modes[resolution] || 30
    commit({
      resolution_preset: resolution,
      width,
      height,
      fps: Math.min(Number(config?.fps || 30), maxFps),
    }, true)
  }, [camera, commit, config])

  const refreshCameras = useCallback(async () => {
    if (runtime.running) {
      showNotice('Stop the camera before detecting modes')
      return
    }
    setCameraBusy(true)
    try {
      const next = await api.cameras(true)
      setCamera(next)
      showNotice(`Detected ${next.cameras.length} camera${next.cameras.length === 1 ? '' : 's'} and ${next.resolutions.length} modes`)
    } catch (error) {
      showNotice(error instanceof Error ? error.message : 'Camera detection failed')
    } finally {
      setCameraBusy(false)
    }
  }, [runtime.running, showNotice])

  const runProviderBenchmark = useCallback(async () => {
    if (dirtyKeys.size) {
      showNotice('Apply pending settings before profiling CPU and CUDA')
      return
    }
    try {
      const status = await api.startTimingBenchmark()
      setTimings((current) => current ? { ...current, benchmark: status } : current)
      showNotice('CPU and CUDA profiling started; the live stream will resume automatically')
    } catch (error) {
      showNotice(error instanceof Error ? error.message : 'Provider profiling could not start')
    }
  }, [dirtyKeys.size, showNotice])

  const uploadSource = useCallback(async (file?: File) => {
    if (!file) return
    setBusy(true)
    try {
      const result = await api.uploadSource(file)
      setConfig((current) => current ? { ...current, source_paths: result.source_paths } : current)
      setDirtyKeys((current) => new Set(current).add('source_paths'))
      showNotice('Source face added; apply to reload the face swapper')
    } catch (error) {
      showNotice(error instanceof Error ? error.message : 'Upload failed')
    } finally {
      setBusy(false)
    }
  }, [showNotice])

  const refreshIdentities = useCallback(async () => {
    const data = await api.identities()
    setIdentityProfiles(data.profiles)
    setIdentityPrompts(data.prompts)
  }, [])

  const startIdentityTrainer = useCallback(async () => {
    if (!identityName.trim()) {
      showNotice('Enter the person\'s name first')
      return
    }
    if (!identityConsent) {
      showNotice('Confirm the person\'s consent before recording')
      return
    }
    setBusy(true)
    try {
      const session = await api.startIdentitySession(identityName.trim(), true, identityMode)
      setIdentitySession(session)
      setPreviewVisible(true)
      showNotice(identityMode === 'dfm' ? 'DFM dataset capture ready; each action records a longer burst' : 'Guided capture ready; follow each action in order')
    } catch (error) {
      showNotice(error instanceof Error ? error.message : 'Could not start identity capture')
    } finally {
      setBusy(false)
    }
  }, [identityConsent, identityMode, identityName, showNotice])

  const cancelIdentityTrainer = useCallback(async () => {
    if (!identitySession) return
    setBusy(true)
    try {
      const result = await api.cancelIdentitySession(identitySession.id)
      setRuntime(result.runtime)
      setIdentitySession(null)
      showNotice('Incomplete identity capture discarded')
    } catch (error) {
      showNotice(error instanceof Error ? error.message : 'Could not cancel identity capture')
    } finally {
      setBusy(false)
    }
  }, [identitySession, showNotice])

  const captureIdentityPrompt = useCallback(async (promptId: string) => {
    if (!identitySession) return
    setIdentityBusyPrompt(promptId)
    try {
      const result = await api.captureIdentityAction(identitySession.id, promptId, identitySession.kind === 'dfm' ? 12 : 1.15)
      setIdentitySession(result.session)
      showNotice(result.accepted > 0
        ? `${result.accepted} clear frames saved · best quality ${result.best_quality ?? '—'}%`
        : result.feedback)
    } catch (error) {
      showNotice(error instanceof Error ? error.message : 'Capture failed')
    } finally {
      setIdentityBusyPrompt(null)
    }
  }, [identitySession, showNotice])

  const finishIdentityTrainer = useCallback(async () => {
    if (!identitySession) return
    setBusy(true)
    try {
      const result = await api.finishIdentitySession(identitySession.id)
      setConfig(result.config)
      setRuntime(result.runtime)
      setIdentitySession(null)
      setIdentityConsent(false)
      setIdentityName('')
      if (result.profile.kind === 'dfm-training-dataset') {
        setIdentityMode('dfm')
        setDfmSelectedProfile(result.profile.id)
        api.dfmStatus(true).then(setDfmStatus).catch(() => undefined)
      }
      setDirtyKeys((current) => {
        const next = new Set(current)
        next.delete('source_paths')
        next.delete('swap_mode')
        return next
      })
      await refreshIdentities()
      showNotice(result.profile.kind === 'dfm-training-dataset'
        ? `${result.profile.name} dataset saved; configure the DFM training workspace below`
        : `${result.profile.name} is now active in the face swapper`)
    } catch (error) {
      showNotice(error instanceof Error ? error.message : 'Could not create identity')
    } finally {
      setBusy(false)
    }
  }, [identitySession, refreshIdentities, showNotice])

  const activateIdentity = useCallback(async (profileId: string) => {
    setBusy(true)
    try {
      const result = await api.activateIdentity(profileId)
      setConfig(result.config)
      setRuntime(result.runtime)
      setDirtyKeys((current) => {
        const next = new Set(current)
        next.delete('source_paths')
        next.delete('swap_mode')
        return next
      })
      await refreshIdentities()
      showNotice(`${result.profile.name} activated without restarting the webcam`)
    } catch (error) {
      showNotice(error instanceof Error ? error.message : 'Could not activate identity')
    } finally {
      setBusy(false)
    }
  }, [refreshIdentities, showNotice])

  const openDfmSetup = useCallback(async () => {
    const environment = dfmStatus?.environment
    if (!environment) return
    setDfmBusyAction('setup')
    try {
      const result = await api.openDfmSetup(environment.distro, dfmEngineRoot || environment.engine_root)
      showNotice(result.command ? `${result.message} Command: ${result.command}` : result.message)
    } catch (error) {
      showNotice(error instanceof Error ? error.message : 'Could not open DFM setup')
    } finally {
      setDfmBusyAction(null)
    }
  }, [dfmEngineRoot, dfmStatus?.environment, showNotice])

  const refreshDfm = useCallback(async () => {
    setDfmBusyAction('refresh')
    try {
      setDfmStatus(await api.dfmStatus(true))
      showNotice('DFM environment refreshed')
    } catch (error) {
      showNotice(error instanceof Error ? error.message : 'Could not refresh DFM environment')
    } finally {
      setDfmBusyAction(null)
    }
  }, [showNotice])

  const prepareDfm = useCallback(async () => {
    const environment = dfmStatus?.environment
    if (!environment || !dfmSelectedProfile) return
    setDfmBusyAction('prepare')
    try {
      const job = await api.prepareDfm(dfmSelectedProfile, environment.distro, dfmEngineRoot || environment.engine_root, dfmBaseWorkspace.trim(), dfmTargetIterations)
      setDfmStatus((current) => current ? { ...current, jobs: [job, ...current.jobs.filter((item) => item.id !== job.id)] } : current)
      showNotice('DFM workspace preparation started')
    } catch (error) {
      showNotice(error instanceof Error ? error.message : 'Could not prepare DFM training')
    } finally {
      setDfmBusyAction(null)
    }
  }, [dfmBaseWorkspace, dfmEngineRoot, dfmSelectedProfile, dfmStatus?.environment, dfmTargetIterations, showNotice])

  const runDfmJobAction = useCallback(async (job: DfmTrainingJob, action: 'extract' | 'train' | 'stop' | 'export' | 'activate') => {
    setDfmBusyAction(`${job.id}:${action}`)
    try {
      if (action === 'activate') {
        const result = await api.activateDfm(job.id)
        setConfig(result.config)
        setRuntime(result.runtime)
        showNotice(`${result.model.name} is now selected as the DFM face model`)
      } else {
        const next = action === 'extract' ? await api.extractDfm(job.id)
          : action === 'train' ? await api.trainDfm(job.id)
            : action === 'stop' ? await api.stopDfm(job.id)
              : await api.exportDfm(job.id)
        setDfmStatus((current) => current ? { ...current, jobs: current.jobs.map((item) => item.id === next.id ? next : item) } : current)
        showNotice(action === 'extract' ? 'Face extraction started'
          : action === 'train' ? 'DFM training started; FaceFlow released its GPU memory'
            : action === 'stop' ? 'Stopping DFM training and saving its last checkpoint'
              : 'DFM export started')
      }
    } catch (error) {
      showNotice(error instanceof Error ? error.message : `Could not ${action} DFM job`)
    } finally {
      setDfmBusyAction(null)
    }
  }, [showNotice])

  const currentModeMaxFps = useMemo(() => {
    if (!camera || !config) return 30
    return camera.modes[String(config.resolution_preset)] || camera.max_fps || 30
  }, [camera, config])

  if (fatalError) {
    return (
      <main className="fatal-screen">
        <CircleAlert size={38} />
        <h1>Could not connect to the FaceFusion API</h1>
        <p>{fatalError}</p>
        <button className="primary-button" onClick={() => window.location.reload()}><RefreshCw size={16} /> Retry</button>
      </main>
    )
  }

  if (!config || !options || !camera) {
    return (
      <main className="loading-screen">
        <div className="brand-mark"><Aperture size={28} /></div>
        <div className="loading-bar"><span /></div>
        <p>Starting FaceFlow and detecting your camera…</p>
      </main>
    )
  }

  const detectorSizes = options.detector_sizes_map[String(config.detector_model)] || ['160x160']
  const swapperSizes = options.face_swapper_sizes[String(config.face_swapper_model)] || ['256x256']
  const provider = String((config.execution_providers || ['cuda'])[0])
  const providerSummary = config.provider_masker === 'tensorrt' || config.provider_deep_swapper === 'tensorrt'
    ? 'TensorRT hybrid'
    : provider.toUpperCase()
  const timingBenchmark = timings?.benchmark
  const timingResult = timingBenchmark?.result
  const cudaPipelineMs = timingResult?.providers.cuda?.pipeline_mean_ms
  const cpuPipelineMs = timingResult?.providers.cpu?.pipeline_mean_ms
  const trainerPrompts = identitySession?.prompts || identityPrompts
  const completedIdentityPrompts = trainerPrompts.filter((prompt) => (identitySession?.prompt_counts[prompt.id] || 0) > 0).length
  const activeCaptureKind = identitySession?.kind || identityMode
  const minimumIdentityFrames = activeCaptureKind === 'dfm' ? 400 : 6
  const minimumIdentityPrompts = activeCaptureKind === 'dfm' ? 5 : 3
  const identityCaptureReady = Boolean(identitySession && identitySession.accepted_frames >= minimumIdentityFrames && completedIdentityPrompts >= minimumIdentityPrompts)
  const dfmEnvironment = dfmStatus?.environment
  const selectedDfmProfile = identityProfiles.find((profile) => profile.id === dfmSelectedProfile)
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><Aperture size={23} /></span>
          <span><strong>FaceFlow</strong><small>Realtime FaceFusion studio</small></span>
        </div>
        <div className="topbar-status">
          <span className={`status-dot ${runtime.capture_active ? 'online' : runtime.error ? 'error' : ''}`} />
          <span>{runtime.capture_active ? 'Camera live' : runtime.running ? 'Starting camera' : runtime.error ? 'Needs attention' : 'Camera off'}</span>
          {runtime.running && <strong>{formatMetric(runtime.processed_fps, ' FPS')}</strong>}
        </div>
        <div className="topbar-actions">
          <button
            className={`topbar-virtual ${config.virtual_cam_enabled ? 'enabled' : ''}`}
            disabled={busy}
            aria-pressed={Boolean(config.virtual_cam_enabled)}
            onClick={() => commit({ virtual_cam_enabled: !config.virtual_cam_enabled })}
            title="Publish processed frames as a virtual webcam"
          >
            <Video size={16} />
            <span>Virtual camera</span>
            <i aria-hidden="true"><i /></i>
          </button>
          <a className="ghost-button api-link" href="/docs" target="_blank" rel="noreferrer">API docs <ChevronRight size={15} /></a>
          {runtime.running ? (
            <button className="stop-button" disabled={busy} onClick={() => handleRuntime('stop')}><VideoOff size={16} /> Stop camera</button>
          ) : (
            <button className="primary-button" disabled={busy} onClick={() => handleRuntime('start')}><Play size={16} fill="currentColor" /> Start camera</button>
          )}
        </div>
      </header>

      <aside className="sidebar">
        <nav>
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon
            return (
              <button key={item.id} className={activeTab === item.id ? 'active' : ''} onClick={() => setActiveTab(item.id)}>
                <Icon size={18} /><span>{item.label}</span>
              </button>
            )
          })}
        </nav>
        <div className="sidebar-card">
          <div className="sidebar-card-icon"><Zap size={17} /></div>
          <div><strong>{config.realtime_fast_analysis ? 'Sweet spot active' : 'Full analysis active'}</strong><small>{providerSummary} + {config.realtime_fast_analysis ? 'Fast live analysis' : 'Refined landmarks'}</small></div>
        </div>
      </aside>

      <main className="workspace">
        <section className="camera-toolbar">
          <div className="camera-picker">
            <span className="toolbar-label">Camera source</span>
            <div className="inline-select">
              <Camera size={17} />
              <select value={config.camera_choice} disabled={cameraBusy || busy} onChange={(event) => changeCamera(event.target.value)}>
                {camera.cameras.map((choice) => <option key={choice}>{choice}</option>)}
              </select>
            </div>
          </div>
          <div className="resolution-group">
            <span className="toolbar-label">Resolution</span>
            <div className="resolution-pills">
              {camera.resolutions.map((resolution) => (
                <button
                  key={resolution}
                  className={config.resolution_preset === resolution ? 'selected' : ''}
                  onClick={() => selectResolution(resolution)}
                  disabled={busy}
                >
                  {resolution}
                  <small>{camera.modes[resolution]} fps</small>
                </button>
              ))}
            </div>
          </div>
          <label className="fps-control">
            <span className="toolbar-label">Capture rate</span>
            <span><input type="number" min="1" max={currentModeMaxFps} value={config.fps} onChange={(event) => stage('fps', Number(event.target.value))} onBlur={() => commit({ fps: Number(config.fps) }, true)} /> FPS</span>
          </label>
          <button className="icon-button" title={runtime.running ? 'Stop the camera before detecting modes' : 'Detect cameras and modes'} onClick={refreshCameras} disabled={cameraBusy || runtime.running}>
            <RefreshCw size={17} className={cameraBusy ? 'spin' : ''} />
          </button>
          <div className="camera-power">
            <span className="toolbar-label">Webcam capture</span>
            {runtime.running ? (
              <button className="stop-button" disabled={busy} onClick={() => handleRuntime('stop')}><VideoOff size={16} /> Stop camera</button>
            ) : (
              <button className="primary-button" disabled={busy} onClick={() => handleRuntime('start')}><Camera size={16} /> Start camera</button>
            )}
          </div>
        </section>

        <div className="studio-grid">
          <section className="preview-column">
            <div className={`preview-card ${runtime.capture_active ? 'is-live' : ''}`}>
              <div className="preview-head">
                <div>
                  <span className="eyebrow">{activeTab === 'trainer' ? 'Raw identity capture' : 'Processed output'}</span>
                  <strong>{activeTab === 'trainer' ? `${config.resolution_preset} · unprocessed camera` : `${config.resolution_preset} · ${config.deep_swapper_model.split('/').pop()?.replaceAll('_', ' ')}`}</strong>
                </div>
                <div className="preview-badges">
                  {config.virtual_cam_enabled && <span className="badge violet"><Video size={13} /> Virtual camera</span>}
                  <span className={`badge ${runtime.capture_active ? 'green' : ''}`}><span className="mini-dot" /> {runtime.capture_active ? 'CAMERA LIVE' : runtime.running ? 'STARTING' : 'CAMERA OFF'}</span>
                  <button className="preview-visibility" onClick={() => setPreviewVisible((visible) => !visible)} title={previewVisible ? 'Hide preview' : 'Show preview'}>
                    {previewVisible ? <EyeOff size={14} /> : <Eye size={14} />}
                    {previewVisible ? 'Hide' : 'Show'}
                  </button>
                </div>
              </div>
              {previewVisible ? <div className="preview-frame">
                {runtime.running && <img src={activeTab === 'trainer' ? '/api/raw-preview.mjpg' : '/api/preview.mjpg'} alt={activeTab === 'trainer' ? 'Raw guided identity capture preview' : 'Processed FaceFusion camera preview'} />}
                {!runtime.has_preview && (
                  <div className="preview-placeholder">
                    <div className="viewfinder"><ScanFace size={48} strokeWidth={1.2} /></div>
                    <strong>{runtime.running ? 'Preparing camera and models' : 'Webcam capture is off'}</strong>
                    <p>{runtime.running ? 'The first frame can take a moment after a model change.' : 'Start the camera when you want FaceFlow to access it.'}</p>
                    {!runtime.running && <button className="primary-button" onClick={() => handleRuntime('start')}><Camera size={16} /> Start camera</button>}
                  </div>
                )}
                {runtime.running && runtime.processed_fps !== undefined && (
                  <div className="fps-overlay"><Activity size={14} /><strong>{runtime.processed_fps.toFixed(1)}</strong><span>processed FPS</span></div>
                )}
              </div> : (
                <div className="preview-hidden">
                  <EyeOff size={28} />
                  <div><strong>Preview hidden</strong><span>Processing and virtual-camera output continue normally.</span></div>
                  <button className="secondary-button" onClick={() => setPreviewVisible(true)}><Eye size={15} /> Show preview</button>
                </div>
              )}
              <div className="preview-footer">
                <p className={runtime.error ? 'error-text' : ''}>{runtime.error || runtime.status.replace('Live performance: ', '')}</p>
              </div>
            </div>

            <div className="metrics-grid">
              <div className="metric-card"><span><Activity size={16} /> Processing</span><strong>{formatMetric(runtime.processed_fps)}</strong><small>frames / second</small></div>
              <div className="metric-card"><span><Zap size={16} /> Model latency</span><strong>{formatMetric(runtime.processing_ms)}</strong><small>milliseconds</small></div>
              <div className="metric-card"><span><Gauge size={16} /> End to end</span><strong>{formatMetric(runtime.end_to_end_ms)}</strong><small>milliseconds</small></div>
              <div className="metric-card"><span><Video size={16} /> Virtual output</span><strong>{config.virtual_cam_enabled ? formatMetric(runtime.virtual_camera_fps) : 'Off'}</strong><small>{runtime.virtual_camera_backend || 'OBS camera'}</small></div>
            </div>

          </section>

          <aside className="inspector">
            <div className="inspector-title">
              <div><span className="eyebrow">Controls</span><h2>{NAV_ITEMS.find((item) => item.id === activeTab)?.label}</h2></div>
              {dirtyKeys.size > 0 && <span className="unsaved-dot">{dirtyKeys.size} unsaved</span>}
            </div>

            {activeTab === 'live' && (
              <div className="panel-stack">
                <Panel title="Face model" description="Choose the transformation applied to every processed frame.">
                  <div className="field-grid">
                    <SelectField label="Swap engine" value={config.swap_mode} options={['deep', 'face', 'none']} onChange={(value) => stage('swap_mode', value)} />
                    {config.swap_mode === 'deep' && (
                      <ModelBrowser
                        models={modelCatalog}
                        value={String(config.deep_swapper_model)}
                        install={modelInstall}
                        disabled={busy}
                        onSelect={selectDeepModel}
                        onInstall={installDeepModel}
                      />
                    )}
                    {config.swap_mode === 'deep' && <RangeField label="Morph strength" value={Number(config.morph)} min={0} max={100} unit="%" onChange={(value) => stage('morph', value)} />}
                    {config.swap_mode === 'face' && <SelectField label="Face swapper model" value={config.face_swapper_model} options={options.face_swapper_models} onChange={(value) => stage('face_swapper_model', value)} />}
                    {config.swap_mode === 'face' && <SelectField label="Pixel boost" value={config.face_swapper_pixel_boost} options={swapperSizes} onChange={(value) => stage('face_swapper_pixel_boost', value)} />}
                  </div>
                  {config.swap_mode === 'face' && (
                    <div className="upload-zone">
                      <Upload size={20} />
                      <span><strong>Add source face</strong><small>{(config.source_paths || []).length} image(s) loaded</small></span>
                      <input type="file" accept="image/*" onChange={(event) => uploadSource(event.target.files?.[0])} />
                      {(config.source_paths || []).length > 0 && <button title="Clear source faces" onClick={async () => { const result = await api.clearSources(); setConfig({ ...config, source_paths: result.source_paths }); setDirtyKeys((current) => new Set(current).add('source_paths')) }}><Trash2 size={16} /></button>}
                    </div>
                  )}
                </Panel>
                <Panel title="Realtime behavior">
                  <Toggle label="GPU colour pipeline" description="Keep DFM buffers and colour matching in VRAM; precise face-edge masks stay on the stable path." checked={Boolean(config.gpu_pipeline_enabled)} onChange={(checked) => stage('gpu_pipeline_enabled', checked)} accent />
                  <Toggle label="Fast live analysis" description="Measured high-FPS path; bypasses analysis modules not needed for live swapping." checked={Boolean(config.realtime_fast_analysis)} onChange={(checked) => stage('realtime_fast_analysis', checked)} accent />
                  {!Boolean(config.realtime_fast_analysis) && (
                    <>
                      <Toggle label="Temporal face tracking" description="Experimental FPS mode. Leave off if detection or alignment flickers." checked={Boolean(config.temporal_face_tracking)} onChange={(checked) => stage('temporal_face_tracking', checked)} />
                      {Boolean(config.temporal_face_tracking) && <RangeField label="Face analysis interval" value={Number(config.temporal_face_interval || 3)} min={1} max={4} unit=" frames" onChange={(value) => stage('temporal_face_interval', value)} />}
                    </>
                  )}
                  <Toggle label="Detection overlay" description="Draw performance information into the output." checked={Boolean(config.show_overlay)} onChange={(checked) => stage('show_overlay', checked)} />
                  <Toggle label="Face boxes" description="Show the detected face boundary." checked={Boolean(config.show_boxes)} onChange={(checked) => stage('show_boxes', checked)} />
                  <Toggle label="Occlusion mask" description="Preserve objects crossing in front of the face." checked={Boolean(config.use_occlusion)} onChange={(checked) => stage('use_occlusion', checked)} />
                  {Boolean(config.use_occlusion) && (
                    <>
                      <Toggle label="Temporal mask reuse" description="Reuse a stable mask for adjacent frames and refresh immediately when the face crop changes." checked={Boolean(config.temporal_occlusion_reuse)} onChange={(checked) => stage('temporal_occlusion_reuse', checked)} accent />
                      {Boolean(config.temporal_occlusion_reuse) && <RangeField label="Mask refresh interval" value={Number(config.temporal_occlusion_interval || 3)} min={1} max={4} unit=" frames" onChange={(value) => stage('temporal_occlusion_interval', value)} />}
                    </>
                  )}
                </Panel>
              </div>
            )}

            {activeTab === 'trainer' && (
              <div className="panel-stack identity-trainer">
                <div className="trainer-mode-tabs" role="tablist" aria-label="Identity creation method">
                  <button role="tab" aria-selected={identityMode === 'instant'} disabled={Boolean(identitySession)} className={identityMode === 'instant' ? 'active' : ''} onClick={() => setIdentityMode('instant')}>
                    <Zap size={15} /><span><strong>Instant identity</strong><small>Ready after a few poses</small></span>
                  </button>
                  <button role="tab" aria-selected={identityMode === 'dfm'} disabled={Boolean(identitySession)} className={identityMode === 'dfm' ? 'active' : ''} onClick={() => setIdentityMode('dfm')}>
                    <Cpu size={15} /><span><strong>Train a DFM</strong><small>Best quality, long training</small></span>
                  </button>
                </div>
                {!identitySession ? (
                  <Panel
                    title={identityMode === 'dfm' ? 'Record a DFM source dataset' : 'Create an instant identity'}
                    description={identityMode === 'dfm'
                      ? 'Record longer bursts across varied poses and expressions. FaceFlow keeps clear frames for DeepFaceLab.'
                      : 'Record several clear angles and expressions, then average them into a reusable face-swap source.'}
                  >
                    <label className="field">
                      <span className="field-label">Person's name</span>
                      <input type="text" value={identityName} maxLength={80} placeholder="Name this identity" onChange={(event) => setIdentityName(event.target.value)} />
                    </label>
                    <label className="identity-consent">
                      <input type="checkbox" checked={identityConsent} onChange={(event) => setIdentityConsent(event.target.checked)} />
                      <span><strong>Consent confirmed</strong><small>The person agreed to this recording and to creation of a reusable face identity.</small></span>
                    </label>
                    <button className="primary-button identity-start" disabled={busy || !identityName.trim() || !identityConsent} onClick={startIdentityTrainer}>
                      <Camera size={17} /> {identityMode === 'dfm' ? 'Begin DFM dataset capture' : 'Begin guided capture'}
                    </button>
                    <div className={`trainer-explainer ${identityMode === 'dfm' ? 'dfm' : ''}`}>
                      {identityMode === 'dfm' ? <HardDrive size={16} /> : <Zap size={16} />}
                      <span>
                        <strong>{identityMode === 'dfm' ? 'Plan for 10–20 minutes of guided recording' : 'Ready immediately'}</strong>
                        <small>{identityMode === 'dfm'
                          ? 'Each action records 12 seconds at up to 12 clear frames/second. Aim for 2,000+ varied frames before training.'
                          : 'This builds an averaged FaceFusion source profile. The complete source dataset is saved too.'}</small>
                      </span>
                    </div>
                  </Panel>
                ) : (
                  <>
                    <div className="trainer-progress-card">
                      <div><span className="eyebrow">Recording {identitySession.name}</span><strong>{completedIdentityPrompts} of {trainerPrompts.length} actions complete</strong></div>
                      <span>{identitySession.accepted_frames.toLocaleString()} / {identitySession.recommended_frames.toLocaleString()} clear frames</span>
                      <div className="trainer-progress"><i style={{ width: `${Math.min(100, identitySession.accepted_frames / Math.max(1, identitySession.recommended_frames) * 100)}%` }} /></div>
                    </div>
                    <Panel title="Follow the actions" description={identitySession.kind === 'dfm' ? 'Each button records a 12-second burst. Keep moving gently within the requested pose for useful variation.' : 'Face the raw preview. Each button records a short burst and keeps only clear, well-exposed frames.'}>
                      <div className="trainer-prompt-list">
                        {trainerPrompts.map((prompt, index) => {
                          const count = identitySession.prompt_counts[prompt.id] || 0
                          const recording = identityBusyPrompt === prompt.id
                          return (
                            <div className={`trainer-prompt ${count > 0 ? 'complete' : ''}`} key={prompt.id}>
                              <span className="trainer-prompt-number">{count > 0 ? <Check size={14} /> : index + 1}</span>
                              <span className="trainer-prompt-copy"><strong>{prompt.title}</strong><small>{prompt.instruction}</small>{count > 0 && <em>{count} frame{count === 1 ? '' : 's'} accepted</em>}</span>
                              <button className={count > 0 ? 'secondary-button' : 'primary-button'} disabled={Boolean(identityBusyPrompt) || busy} onClick={() => captureIdentityPrompt(prompt.id)}>
                                {recording ? <RefreshCw className="spin" size={15} /> : <Aperture size={15} />}{recording ? (identitySession.kind === 'dfm' ? 'Recording 12s' : 'Recording') : count > 0 ? 'Again' : 'Record'}
                              </button>
                            </div>
                          )
                        })}
                      </div>
                    </Panel>
                    <div className="trainer-actions">
                      <button className="secondary-button" disabled={busy || Boolean(identityBusyPrompt)} onClick={cancelIdentityTrainer}>Discard capture</button>
                      <button className="primary-button trainer-finish" disabled={busy || Boolean(identityBusyPrompt) || !identityCaptureReady} onClick={finishIdentityTrainer}>
                        <Sparkles size={17} /> {identitySession.kind === 'dfm' ? 'Save DFM training dataset' : 'Create and use this identity'}
                      </button>
                    </div>
                    {!identityCaptureReady && <p className="trainer-requirement">{identitySession.kind === 'dfm' ? 'At least 400 clear frames across 5 actions are required; around 2,000 varied frames is strongly recommended.' : 'At least 6 clear frames across 3 actions are required. Completing all actions gives the best result.'}</p>}
                  </>
                )}

                {!identitySession && (
                  <Panel title="Saved identities" description="Switch identities live without reopening the physical webcam.">
                    {identityProfiles.length > 0 ? <div className="identity-profile-grid">
                      {identityProfiles.map((profile) => (
                        <div className={`identity-profile ${profile.active ? 'active' : ''}`} key={profile.id}>
                          <img src={profile.thumbnail_url} alt={profile.name} />
                          <span><strong>{profile.name}</strong><small>{profile.kind === 'dfm-training-dataset' ? 'DFM dataset' : `${profile.source_count} averaged views`} · {profile.frame_count.toLocaleString()} frames</small></span>
                          <div>
                            <button className={profile.active ? 'secondary-button' : 'primary-button'} disabled={busy || profile.active} onClick={() => activateIdentity(profile.id)}>{profile.active ? <Check size={14} /> : <Play size={14} />}{profile.active ? 'Active' : 'Use'}</button>
                            <a className="icon-button" href={profile.dataset_url} title="Download complete training dataset"><Download size={15} /></a>
                          </div>
                        </div>
                      ))}
                    </div> : <div className="model-empty"><UserRoundPlus size={22} /><strong>No saved identities yet</strong><span>Your first guided capture will appear here.</span></div>}
                  </Panel>
                )}

                {!identitySession && identityMode === 'dfm' && (
                  <>
                    <Panel title="DFM training engine" description="DeepFaceLab SAEHD runs separately in WSL2 so it can use the RTX 5080-compatible TensorFlow build.">
                      {dfmEnvironment ? (
                        <div className={`dfm-engine-state ${dfmEnvironment.ready ? 'ready' : 'missing'}`}>
                          <span>{dfmEnvironment.ready ? <Check size={17} /> : <CircleAlert size={17} />}</span>
                          <div>
                            <strong>{dfmEnvironment.ready ? 'DeepFaceLab is ready' : 'DeepFaceLab setup required'}</strong>
                            <small>{dfmEnvironment.distro || 'No Ubuntu distribution'}{dfmEnvironment.distro_versions[dfmEnvironment.distro] ? ` ${dfmEnvironment.distro_versions[dfmEnvironment.distro]}` : ''} · {dfmEnvironment.gpu || 'GPU not checked'}</small>
                          </div>
                        </div>
                      ) : <div className="dfm-engine-state"><RefreshCw className="spin" size={17} /><span>Checking the DFM environment…</span></div>}
                      <label className="field">
                        <span className="field-label">DeepFaceLab root inside WSL</span>
                        <input value={dfmEngineRoot} placeholder="/home/user/DeepFaceLab-RTX5000" onChange={(event) => setDfmEngineRoot(event.target.value)} />
                      </label>
                      <div className="dfm-engine-actions">
                        {!dfmEnvironment?.ready && <button className="primary-button" disabled={!dfmEnvironment || Boolean(dfmBusyAction)} onClick={openDfmSetup}><Download size={15} /> Open compatible setup</button>}
                        <button className="secondary-button" disabled={Boolean(dfmBusyAction)} onClick={refreshDfm}><RefreshCw className={dfmBusyAction === 'refresh' ? 'spin' : ''} size={15} /> Refresh</button>
                      </div>
                      {!dfmEnvironment?.ready && <p className="dfm-note">Setup uses the maintained RTX 5000 WSL fork and may request your Linux password. FaceFlow will not modify the older Ubuntu 22.04 environment.</p>}
                    </Panel>

                    <Panel title="Prepare training" description="A real-time DFM starts from an RTT/pretrained SAEHD workspace containing a broad destination faceset and model checkpoint.">
                      <label className="field">
                        <span className="field-label">Source identity</span>
                        <select value={dfmSelectedProfile} onChange={(event) => setDfmSelectedProfile(event.target.value)}>
                          <option value="">Choose an identity dataset</option>
                          {identityProfiles.map((profile) => <option key={profile.id} value={profile.id}>{profile.name} · {profile.frame_count.toLocaleString()} frames</option>)}
                        </select>
                      </label>
                      {selectedDfmProfile && !selectedDfmProfile.dfm_ready && <div className="dfm-warning"><CircleAlert size={15} /><span>This identity has only {selectedDfmProfile.frame_count} frames. Record a DFM dataset first; a few instant-profile images cannot train a healthy model.</span></div>}
                      <label className="field">
                        <span className="field-label">RTT base workspace inside WSL</span>
                        <input value={dfmBaseWorkspace} placeholder="/home/user/RTT-224/workspace" onChange={(event) => setDfmBaseWorkspace(event.target.value)} />
                        <small>Must contain <code>data_dst/aligned</code> and a pretrained <code>model</code> folder.</small>
                      </label>
                      <label className="field">
                        <span className="field-label">Iteration progress marker</span>
                        <select value={dfmTargetIterations} onChange={(event) => setDfmTargetIterations(Number(event.target.value))}>
                          <option value={250000}>250,000 · quick experiment</option>
                          <option value={500000}>500,000 · first useful checkpoint</option>
                          <option value={1000000}>1,000,000 · recommended</option>
                          <option value={2000000}>2,000,000 · high quality</option>
                        </select>
                      </label>
                      <button className="primary-button identity-start" disabled={!dfmEnvironment?.ready || !selectedDfmProfile?.dfm_ready || !dfmBaseWorkspace.trim() || Boolean(dfmBusyAction)} onClick={prepareDfm}>
                        {dfmBusyAction === 'prepare' ? <RefreshCw className="spin" size={16} /> : <HardDrive size={16} />} Prepare DFM workspace
                      </button>
                      <p className="dfm-note">Training stops FaceFlow capture and frees its inference sessions so DeepFaceLab gets the full 16 GB VRAM. The virtual camera remains advertised with its standby frame.</p>
                    </Panel>

                    {dfmStatus && dfmStatus.jobs.length > 0 && (
                      <Panel title="Training jobs" description="Extraction, training and export are resumable from the saved WSL workspace.">
                        <div className="dfm-job-list">
                          {dfmStatus.jobs.map((job) => {
                            const actionBusy = Boolean(dfmBusyAction?.startsWith(`${job.id}:`))
                            const activeJob = ['preparing', 'extracting', 'training', 'stopping', 'exporting'].includes(job.phase)
                            return (
                              <div className={`dfm-job ${job.phase}`} key={job.id}>
                                <div className="dfm-job-head"><span><strong>{job.name}</strong><small>{job.phase.replace('-', ' ')} · {job.aligned_frames.toLocaleString()} aligned faces</small></span><em>{Math.round(job.progress * 100)}%</em></div>
                                <div className="dfm-job-progress"><i style={{ width: `${Math.max(job.current_iteration ? 1 : 0, job.progress * 100)}%` }} /></div>
                                <div className="dfm-job-stats"><span>{job.current_iteration.toLocaleString()} iterations</span><span>marker {job.target_iterations.toLocaleString()}</span></div>
                                {job.error && <div className="dfm-warning"><CircleAlert size={14} /><span>{job.error}</span></div>}
                                {job.log_tail && <details><summary>Latest DeepFaceLab output</summary><pre>{job.log_tail}</pre></details>}
                                <div className="dfm-job-actions">
                                  {job.phase === 'prepared' && <button className="primary-button" disabled={actionBusy} onClick={() => runDfmJobAction(job, 'extract')}><Aperture size={14} /> Extract faces</button>}
                                  {['ready', 'stopped'].includes(job.phase) && <button className="primary-button" disabled={actionBusy} onClick={() => runDfmJobAction(job, 'train')}><Play size={14} /> {job.current_iteration ? 'Resume training' : 'Start training'}</button>}
                                  {['training', 'extracting', 'exporting'].includes(job.phase) && <button className="stop-button" disabled={actionBusy} onClick={() => runDfmJobAction(job, 'stop')}><Square size={13} /> Stop safely</button>}
                                  {['ready', 'stopped', 'error'].includes(job.phase) && job.current_iteration > 0 && <button className="secondary-button" disabled={actionBusy} onClick={() => runDfmJobAction(job, 'export')}><Download size={14} /> Export DFM</button>}
                                  {job.phase === 'complete' && <button className="primary-button" disabled={actionBusy} onClick={() => runDfmJobAction(job, 'activate')}><Play size={14} /> Use this DFM</button>}
                                  {activeJob && <span className="dfm-job-running"><RefreshCw className="spin" size={13} /> Working</span>}
                                </div>
                              </div>
                            )
                          })}
                        </div>
                      </Panel>
                    )}
                  </>
                )}
              </div>
            )}

            {activeTab === 'camera' && (
              <div className="panel-stack">
                <Panel title="Capture pipeline" description={`${camera.camera_name} reports ${camera.resolutions.length} modes, up to ${camera.max_fps} FPS.`}>
                  <div className="field-grid two-columns">
                    <SelectField label="Backend" value={config.backend} options={options.backends} onChange={(value) => stage('backend', value)} />
                    <SelectField label="Codec / FOURCC" value={config.force_fourcc} options={options.fourcc} onChange={(value) => stage('force_fourcc', value)} />
                    <SelectField label="Color handling" value={config.color_mode} options={['Auto (BGR->RGB)', 'Assume RGB (no swap)']} onChange={(value) => stage('color_mode', value)} />
                    <RangeField label="Black-frame retries" value={Number(config.retry_black)} min={0} max={10} onChange={(value) => stage('retry_black', value)} />
                  </div>
                  <Toggle label="Driver RGB conversion" checked={Boolean(config.convert_rgb)} onChange={(checked) => stage('convert_rgb', checked)} />
                  <Toggle label="Gentle reconnect" description="Avoid aggressive reopen loops when the camera is slow." checked={Boolean(config.gentle_mode)} onChange={(checked) => stage('gentle_mode', checked)} />
                  <Toggle label="Automatic capture repair" checked={Boolean(config.auto_repair)} onChange={(checked) => stage('auto_repair', checked)} />
                </Panel>
                <Panel title="Camera image controls">
                  <Toggle label="Lock exposure" description="Manual mode can trigger edge halos on some USB camera drivers; automatic exposure is recommended." checked={Boolean(config.lock_exposure)} onChange={(checked) => stage('lock_exposure', checked)} />
                  <RangeField label="Exposure" value={Number(config.exposure_value)} min={-13} max={-1} step={0.5} onChange={(value) => stage('exposure_value', value)} />
                  <Toggle label="Lock white balance" checked={Boolean(config.lock_wb)} onChange={(checked) => stage('lock_wb', checked)} />
                  <RangeField label="White balance" value={Number(config.wb_temperature)} min={2800} max={6500} step={100} unit=" K" onChange={(value) => stage('wb_temperature', value)} />
                  <Toggle label="Native OpenCV window" description="Separate full-rate desktop preview for diagnostics." checked={Boolean(config.show_native)} onChange={(checked) => stage('show_native', checked)} />
                </Panel>
              </div>
            )}

            {activeTab === 'processing' && (
              <div className="panel-stack">
                <Panel title="Face detection">
                  <div className="field-grid two-columns">
                    <SelectField label="Detector" value={config.detector_model} options={options.detector_models} onChange={(value) => { stage('detector_model', value); stage('detector_size', options.detector_sizes_map[value]?.[0] || '160x160') }} />
                    <SelectField label="Detector size" value={config.detector_size} options={detectorSizes} onChange={(value) => stage('detector_size', value)} />
                    <SelectField label="Landmarker" value={config.landmarker_model} options={options.landmarker_models} onChange={(value) => stage('landmarker_model', value)} />
                    <SelectField label="Selector mode" value={config.selector_mode} options={options.selector_modes} onChange={(value) => stage('selector_mode', value)} />
                    <SelectField label="Occluder" value={config.occluder_model} options={options.occluder_models} onChange={(value) => stage('occluder_model', value)} />
                    <SelectField label="Parser" value={config.parser_model} options={options.parser_models} onChange={(value) => stage('parser_model', value)} />
                  </div>
                  <RangeField label="Detection confidence" value={Number(config.detector_score)} min={0} max={1} step={0.05} onChange={(value) => stage('detector_score', value)} />
                  <Toggle label="Detector fallback" description="Try another detector after an extended no-face streak." checked={Boolean(config.auto_fallback)} onChange={(checked) => stage('auto_fallback', checked)} />
                </Panel>
                <Panel title="Enhancers" description="These add quality, but usually reduce realtime throughput.">
                  <Toggle label="Face enhancer" checked={Boolean(config.face_enhancer_enabled)} onChange={(checked) => stage('face_enhancer_enabled', checked)} />
                  {config.face_enhancer_enabled && <SelectField label="Face enhancer model" value={config.face_enhancer_model} options={options.face_enhancer_models} onChange={(value) => stage('face_enhancer_model', value)} />}
                  <Toggle label="Frame enhancer" checked={Boolean(config.frame_enhancer_enabled)} onChange={(checked) => stage('frame_enhancer_enabled', checked)} />
                  {config.frame_enhancer_enabled && <SelectField label="Frame enhancer model" value={config.frame_enhancer_model} options={options.frame_enhancer_models} onChange={(value) => stage('frame_enhancer_model', value)} />}
                  {(config.face_enhancer_enabled || config.frame_enhancer_enabled) && <Toggle label="Asynchronous enhancement" checked={Boolean(config.enhance_async)} onChange={(checked) => stage('enhance_async', checked)} accent />}
                </Panel>
                <Panel title="Optional processors">
                  <Toggle label="Frame colorizer" checked={Boolean(config.frame_colorizer_enabled)} onChange={(checked) => stage('frame_colorizer_enabled', checked)} />
                  <Toggle label="Expression restorer" checked={Boolean(config.expression_restorer_enabled)} onChange={(checked) => stage('expression_restorer_enabled', checked)} />
                  <Toggle label="Age modifier" checked={Boolean(config.age_modifier_enabled)} onChange={(checked) => stage('age_modifier_enabled', checked)} />
                  <Toggle label="Face editor" checked={Boolean(config.face_editor_enabled)} onChange={(checked) => stage('face_editor_enabled', checked)} />
                  <Toggle label="Lip syncer" checked={Boolean(config.lip_syncer_enabled)} onChange={(checked) => stage('lip_syncer_enabled', checked)} />
                </Panel>
              </div>
            )}

            {activeTab === 'routing' && (
              <div className="panel-stack">
                <div className="sweetspot-callout">
                  <span><Zap size={18} /></span>
                  <div><strong>Stable sweet spot: CUDA masks + TensorRT face model</strong><p>XSeg stays on CUDA for clean mask edges; DFM models use accuracy-safe TensorRT FP32. Mask FP16 remains available only for experiments.</p></div>
                </div>
                <Panel title="CPU vs CUDA profiler" description="Runs the currently applied models at 1280x720. Model loading and warm-up are excluded.">
                  <div className="timing-benchmark-toolbar">
                    <button className="secondary-button timing-run-button" disabled={Boolean(timingBenchmark?.running)} onClick={runProviderBenchmark}>
                      <Activity className={timingBenchmark?.running ? 'spin' : ''} size={16} />
                      {timingBenchmark?.running ? 'Profiling…' : timingResult ? 'Run again' : 'Benchmark CPU + CUDA'}
                    </button>
                    <div className="timing-benchmark-state">
                      <strong>{timingBenchmark?.phase || 'Ready to profile'}</strong>
                      <span>{timingBenchmark?.running ? `${Math.round(timingBenchmark.progress * 100)}%` : timingResult ? `${timingResult.frames_per_provider} measured frames per provider` : 'No comparison recorded yet'}</span>
                    </div>
                  </div>
                  {timingBenchmark?.running && <div className="timing-progress"><span style={{ width: `${Math.max(2, timingBenchmark.progress * 100)}%` }} /></div>}
                  {timingBenchmark?.error && <div className="timing-error"><CircleAlert size={15} />{timingBenchmark.error}</div>}
                  {timingResult && (
                    <div className="pipeline-timing-grid">
                      <div><small>Whole pipeline · CUDA</small><strong>{formatTiming(cudaPipelineMs)}</strong></div>
                      <div><small>Whole pipeline · CPU</small><strong>{formatTiming(cpuPipelineMs)}</strong></div>
                      <div><small>Current measured model</small><strong>{timingResult.model || timingResult.swap_mode || 'Current settings'}</strong></div>
                    </div>
                  )}
                </Panel>
                <Panel title="Module providers" description="Each figure is the combined inference cost for that module in one processed frame. Disabled or input-dependent modules stay unmeasured until they actually run.">
                  <div className="routing-grid">
                    {options.execution_routes.map((route) => (
                      <RoutingField
                        key={route.key}
                        label={route.label}
                        value={String(config[route.key])}
                        providers={options.execution_providers}
                        comparison={timingResult?.comparisons[route.key]}
                        liveTiming={timings?.live.modules[route.key]?.providers}
                        onChange={(value) => stage(route.key, value)}
                      />
                    ))}
                  </div>
                </Panel>
              </div>
            )}

            {activeTab === 'system' && (
              <div className="panel-stack">
                <Panel title="Execution runtime">
                  <div className="field-grid two-columns">
                    <SelectField label="Primary provider" value={provider} options={options.execution_providers} onChange={(value) => stage('execution_providers', [value])} />
                    <label className="field"><span className="field-label">Device ID</span><input type="text" value={String((config.execution_device_ids || ['0'])[0])} onChange={(event) => stage('execution_device_ids', [event.target.value])} /></label>
                    <SelectField label="VRAM strategy" value={config.video_memory_strategy} options={options.video_memory_strategies} onChange={(value) => stage('video_memory_strategy', value)} />
                  </div>
                  <Toggle label="Fast startup" description="Skip redundant model checks when local checkpoints are already present." checked={Boolean(config.fast_startup)} onChange={(checked) => stage('fast_startup', checked)} />
                </Panel>
                <Panel title="Maintenance">
                  <div className="maintenance-actions">
                    <button className="secondary-button" onClick={() => handleRuntime('restart')}><RotateCcw size={16} /> Restart stream</button>
                    <button className="secondary-button" onClick={async () => { setBusy(true); const result = await api.freeVram(); setRuntime(result.runtime); setBusy(false); showNotice('Inference pools rebuilt') }}><HardDrive size={16} /> Free VRAM</button>
                    <a className="secondary-button" href="/docs" target="_blank" rel="noreferrer"><SlidersHorizontal size={16} /> Open API docs</a>
                  </div>
                </Panel>
                <div className="api-callout">
                  <Cpu size={22} />
                  <div><strong>Local API is ready</strong><p>FastAPI exposes camera discovery, settings, lifecycle, benchmark, preview and source-upload endpoints at <code>/api</code>.</p></div>
                </div>
              </div>
            )}
          </aside>
        </div>
      </main>

      {dirtyKeys.size > 0 && (
        <div className="apply-bar">
          <div><span className="unsaved-pulse" /><strong>{dirtyKeys.size} setting{dirtyKeys.size === 1 ? '' : 's'} changed</strong><small>The active stream keeps its current configuration until you apply.</small></div>
          <button className="ghost-button" onClick={() => { api.bootstrap().then((data) => { setConfig(data.config); setDirtyKeys(new Set()) }) }}>Discard</button>
          <button className="primary-button" disabled={busy} onClick={applyPending}><Check size={16} /> Apply {runtime.running && [...dirtyKeys].some((key) => CAPTURE_RESTART_KEYS.has(key)) ? '& restart source' : runtime.running ? 'live' : 'changes'}</button>
        </div>
      )}

      {notice && <div className="toast"><Check size={16} />{notice}</div>}
    </div>
  )
}

export default App
