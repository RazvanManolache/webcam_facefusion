import type { Bootstrap, CameraInfo, Config, RuntimeStatus, Benchmark, TimingsPayload, ProviderBenchmarkStatus, ModelCatalogPayload, ModelInstallStatus, IdentityCatalog, IdentitySession, IdentityCaptureResult, IdentityProfile, DfmTrainingJob, DfmTrainingStatus, DeepModelInfo } from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
      ...init?.headers,
    },
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(body.detail || `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

export const api = {
  bootstrap: () => request<Bootstrap>('/api/bootstrap'),
  config: () => request<Config>('/api/config'),
  status: () => request<RuntimeStatus>('/api/status'),
  cameras: (refresh = false, cameraChoice?: string) => {
    const query = new URLSearchParams({ refresh: String(refresh) })
    if (cameraChoice) query.set('camera_choice', cameraChoice)
    return request<CameraInfo>(`/api/cameras?${query.toString()}`)
  },
  patchConfig: (values: Partial<Config>, restart = false, clearModels = false) =>
    request<{ config: Config; runtime: RuntimeStatus }>('/api/config', {
      method: 'PATCH',
      body: JSON.stringify({ values, restart, clear_models: clearModels }),
    }),
  start: () => request<RuntimeStatus>('/api/runtime/start', { method: 'POST' }),
  stop: () => request<RuntimeStatus>('/api/runtime/stop', { method: 'POST' }),
  startCapture: () => request<RuntimeStatus>('/api/capture/start', { method: 'POST' }),
  stopCapture: () => request<RuntimeStatus>('/api/capture/stop', { method: 'POST' }),
  restart: (clearModels = false) => request<RuntimeStatus>('/api/runtime/restart', {
    method: 'POST',
    body: JSON.stringify({ clear_models: clearModels }),
  }),
  freeVram: () => request<{ ok: boolean; runtime: RuntimeStatus }>('/api/runtime/free-vram', { method: 'POST' }),
  benchmark: () => request<{ ok: boolean; message: string }>('/api/benchmark', { method: 'POST' }),
  latestBenchmark: () => request<Benchmark>('/api/benchmark/latest'),
  timings: () => request<TimingsPayload>('/api/timings'),
  startTimingBenchmark: () => request<ProviderBenchmarkStatus>('/api/timings/benchmark', { method: 'POST' }),
  models: () => request<ModelCatalogPayload>('/api/models'),
  installModel: (modelId: string) => request<ModelInstallStatus>('/api/models/install', {
    method: 'POST',
    body: JSON.stringify({ model_id: modelId }),
  }),
  uploadSource: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<{ path: string; source_paths: string[] }>('/api/source', { method: 'POST', body: form })
  },
  clearSources: () => request<{ source_paths: string[] }>('/api/source', { method: 'DELETE' }),
  identities: () => request<IdentityCatalog>('/api/identities'),
  startIdentitySession: (name: string, consentConfirmed: boolean, kind: 'instant' | 'dfm' = 'instant') => request<IdentitySession>('/api/identities/session', {
    method: 'POST',
    body: JSON.stringify({ name, consent_confirmed: consentConfirmed, kind }),
  }),
  captureIdentityAction: (sessionId: string, promptId: string, durationSeconds = 1.15) => request<IdentityCaptureResult>(`/api/identities/session/${encodeURIComponent(sessionId)}/capture`, {
    method: 'POST',
    body: JSON.stringify({ prompt_id: promptId, duration_seconds: durationSeconds }),
  }),
  cancelIdentitySession: (sessionId: string) => request<{ ok: boolean; runtime: RuntimeStatus }>(`/api/identities/session/${encodeURIComponent(sessionId)}`, { method: 'DELETE' }),
  finishIdentitySession: (sessionId: string) => request<{ ok: boolean; profile: IdentityProfile; config: Config; runtime: RuntimeStatus }>(`/api/identities/session/${encodeURIComponent(sessionId)}/finish`, { method: 'POST' }),
  activateIdentity: (profileId: string) => request<{ ok: boolean; profile: IdentityProfile; config: Config; runtime: RuntimeStatus }>(`/api/identities/${encodeURIComponent(profileId)}/activate`, { method: 'POST' }),
  dfmStatus: (refresh = false) => request<DfmTrainingStatus>(`/api/dfm?refresh=${String(refresh)}`),
  openDfmSetup: (distro: string, engineRoot: string) => request<{ launched: boolean; needs_distro: boolean; command?: string; message: string }>('/api/dfm/setup', {
    method: 'POST',
    body: JSON.stringify({ distro, engine_root: engineRoot }),
  }),
  prepareDfm: (profileId: string, distro: string, engineRoot: string, baseWorkspace: string, targetIterations: number) => request<DfmTrainingJob>('/api/dfm/jobs', {
    method: 'POST',
    body: JSON.stringify({ profile_id: profileId, distro, engine_root: engineRoot, base_workspace: baseWorkspace, target_iterations: targetIterations }),
  }),
  extractDfm: (jobId: string) => request<DfmTrainingJob>(`/api/dfm/jobs/${encodeURIComponent(jobId)}/extract`, { method: 'POST' }),
  trainDfm: (jobId: string) => request<DfmTrainingJob>(`/api/dfm/jobs/${encodeURIComponent(jobId)}/train`, { method: 'POST' }),
  stopDfm: (jobId: string) => request<DfmTrainingJob>(`/api/dfm/jobs/${encodeURIComponent(jobId)}/stop`, { method: 'POST' }),
  exportDfm: (jobId: string) => request<DfmTrainingJob>(`/api/dfm/jobs/${encodeURIComponent(jobId)}/export`, { method: 'POST' }),
  activateDfm: (jobId: string) => request<{ ok: boolean; model: DeepModelInfo; config: Config; runtime: RuntimeStatus; job: DfmTrainingJob }>(`/api/dfm/jobs/${encodeURIComponent(jobId)}/activate`, { method: 'POST' }),
}
