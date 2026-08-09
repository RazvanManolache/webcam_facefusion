export type Config = Record<string, any>

export interface RuntimeStatus {
  running: boolean
  capture_active: boolean
  status: string
  error?: string | null
  preview_sequence: number
  has_preview: boolean
  uptime_seconds: number
  start_reason?: 'manual' | 'demand' | 'trainer' | null
  processing_mode?: 'none' | 'last' | null
  preview_clients?: number
  demand_monitor_enabled?: boolean
  external_camera_demand?: boolean
  virtual_camera_advertised?: boolean
  virtual_camera_streaming?: boolean
  virtual_camera_error?: string | null
  processed_fps?: number
  processing_ms?: number
  end_to_end_ms?: number
  camera_frames_skipped_percent?: number
  preview_fps?: number
  virtual_camera_fps?: number
  virtual_camera_resolution?: string
  virtual_camera_backend?: string
  benchmark_running?: boolean
  benchmark_remaining_seconds?: number
  gpu_pipeline?: {
    available: boolean
    enabled: boolean
    active: boolean
    mean_ms?: number | null
    samples: number
    failures: number
    last_error?: string | null
  }
}

export interface CameraInfo {
  cameras: string[]
  selected: string
  camera_name: string
  modes: Record<string, number>
  resolutions: string[]
  max_fps: number
}

export interface Options {
  detector_models: string[]
  detector_sizes_map: Record<string, string[]>
  landmarker_models: string[]
  occluder_models: string[]
  parser_models: string[]
  deep_models: string[]
  deep_model_catalog: DeepModelInfo[]
  face_swapper_models: string[]
  frame_enhancer_models: string[]
  face_enhancer_models: string[]
  execution_providers: string[]
  execution_routes: Array<{ key: string; label: string }>
  selector_modes: string[]
  frame_colorizer_models: string[]
  frame_colorizer_sizes: string[]
  expression_restorer_models: string[]
  expression_restorer_areas: string[]
  age_modifier_models: string[]
  face_editor_models: string[]
  face_debugger_items: string[]
  lip_syncer_models: string[]
  face_swapper_sizes: Record<string, string[]>
  backends: string[]
  fourcc: string[]
  video_memory_strategies: string[]
}

export interface DeepModelInfo {
  id: string
  name: string
  gender: 'male' | 'female' | 'other'
  resolution?: number | null
  creator: string
  installed: boolean
  downloadable: boolean
  size_bytes?: number | null
  thumbnail_url: string
  hub_url?: string | null
}

export interface ModelInstallStatus {
  running: boolean
  model_id?: string | null
  progress: number
  downloaded_bytes: number
  total_bytes: number
  error?: string | null
  complete: boolean
}

export interface ModelCatalogPayload {
  models: DeepModelInfo[]
  install: ModelInstallStatus
}

export interface Bootstrap {
  config: Config
  options: Options
  camera: CameraInfo
  runtime: RuntimeStatus
}

export interface Benchmark {
  resolution: string
  camera_fps: number
  duration_seconds: number
  processed_frames: number
  dropped_camera_frames: number
  processed_fps: number
  mean_processing_latency_ms: number
  p95_processing_latency_ms: number
  frames_with_face: number
  valid_face_benchmark: boolean
  virtual_camera_enabled?: boolean
  virtual_camera_frames_sent?: number
  virtual_camera_backend?: string | null
  virtual_camera_error?: string | null
}

export interface TimingStats {
  mean_ms: number
  median_ms: number
  p95_ms: number
  samples: number
}

export interface ModuleProviderTiming {
  per_call?: TimingStats | null
  per_frame?: TimingStats | null
  models: Record<string, TimingStats | null>
  age_seconds?: number | null
}

export interface ModuleTiming {
  label: string
  providers: Record<string, ModuleProviderTiming>
}

export interface TimingSummary {
  generated_at: number
  modules: Record<string, ModuleTiming>
}

export interface ProviderTimingComparison {
  label: string
  cuda?: ModuleProviderTiming | null
  cpu?: ModuleProviderTiming | null
  cuda_mean_ms?: number | null
  cpu_mean_ms?: number | null
  cpu_to_cuda_ratio?: number | null
  active: boolean
}

export interface ProviderBenchmarkResult {
  created_at: number
  machine: string
  resolution: string
  model?: string | null
  swap_mode?: string | null
  forced_full_analysis: boolean
  frames_per_provider: number
  providers: Record<string, {
    provider: string
    frames: number
    pipeline_mean_ms: number
    pipeline_median_ms: number
    pipeline_p95_ms: number
    modules: Record<string, ModuleTiming>
  }>
  comparisons: Record<string, ProviderTimingComparison>
}

export interface ProviderBenchmarkStatus {
  running: boolean
  phase: string
  progress: number
  error?: string | null
  started_at?: number | null
  result?: ProviderBenchmarkResult | null
}

export interface TimingsPayload {
  live: TimingSummary
  benchmark: ProviderBenchmarkStatus
}

export interface IdentityPrompt {
  id: string
  title: string
  instruction: string
}

export interface IdentitySession {
  id: string
  name: string
  created_at: number
  consent_confirmed: boolean
  capturing: boolean
  kind: 'instant' | 'dfm'
  accepted_frames: number
  recommended_frames: number
  prompt_counts: Record<string, number>
  prompts: IdentityPrompt[]
}

export interface IdentityProfile {
  id: string
  name: string
  created_at: number
  frame_count: number
  kind: 'averaged-face-source' | 'dfm-training-dataset'
  dfm_ready: boolean
  source_count: number
  prompts_completed: string[]
  thumbnail_url: string
  dataset_url: string
  active: boolean
}

export interface IdentityCatalog {
  profiles: IdentityProfile[]
  prompts: IdentityPrompt[]
}

export interface IdentityCaptureResult {
  session: IdentitySession
  prompt_id: string
  sampled: number
  accepted: number
  best_quality?: number | null
  feedback: string
}

export interface DfmEnvironment {
  wsl_available: boolean
  distros: string[]
  distro_versions: Record<string, string>
  recommended_distro: string
  distro: string
  user: string
  gpu: string
  engine_root: string
  repository_present: boolean
  setup_present: boolean
  ready: boolean
  error?: string | null
  repository: string
}

export interface DfmTrainingJob {
  id: string
  profile_id: string
  name: string
  distro: string
  engine_root: string
  base_workspace: string
  workspace: string
  source_frames: number
  aligned_frames: number
  target_iterations: number
  current_iteration: number
  progress: number
  phase: 'preparing' | 'prepared' | 'extracting' | 'ready' | 'training' | 'stopping' | 'stopped' | 'exporting' | 'complete' | 'error'
  running: boolean
  error?: string | null
  model_id?: string | null
  log_tail: string
  created_at: number
  updated_at: number
}

export interface DfmTrainingStatus {
  environment: DfmEnvironment
  jobs: DfmTrainingJob[]
}
