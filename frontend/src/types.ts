export interface Telemetry {
  frame: number
  scenario: string
  simulation_time_s: number
  target_world_m: [number, number, number]
  opencv_pixel: [number, number] | null
  kalman_pixel: [number, number] | null
  tracking_error_px: number | null
  camera_pan_rad: number
  camera_tilt_rad: number
  fov_deg?: [number, number]
  locked: boolean
  acquisition_time_s: number | null
  lock_retention_pct: number
  trajectory_m: [number, number, number][]
}
