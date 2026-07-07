import "./style.css";
import { inject } from "@vercel/analytics";
import vertexSource from "./shaders/fullscreen.vert.glsl?raw";
import fragmentSource from "./shaders/sunset.frag.glsl?raw";

/**
 * Minimal WebGL2 harness for a single-pass fullscreen fragment shader.
 *
 * The entire image is produced by the fragment shader; the harness supplies
 * a fullscreen triangle (generated from gl_VertexID, no buffers), resolution
 * and time uniforms, and a cinematic damped orbit camera (yaw/pitch).
 */

const MAX_DEVICE_PIXEL_RATIO = 2;

/**
 * Cinematic orbit controller.
 *
 * Dragging sets *target* angles; the rendered angles chase the targets with
 * a frame-rate-independent exponential spring, so the camera feels heavy and
 * damped rather than immediate. On release, residual pointer velocity keeps
 * pushing the target briefly (inertia) and decays exponentially.
 *
 * Yaw rotates strictly about the world Y axis and pitch is clamped, so the
 * horizon always stays level and in frame.
 */
class OrbitCamera {
  private targetYaw = 0;
  private targetPitch = degToRad(11.5);
  yaw = this.targetYaw;
  pitch = this.targetPitch;

  private dragging = false;
  private lastX = 0;
  private lastY = 0;
  private velYaw = 0;   // rad/s, sampled while dragging, becomes inertia
  private velPitch = 0;
  private lastMoveTime = 0;

  private static readonly PITCH_MIN = degToRad(-2);
  private static readonly PITCH_MAX = degToRad(25);
  private static readonly SMOOTHING = 3.2;      // spring stiffness (1/s)
  private static readonly INERTIA_DECAY = 2.4;  // inertia half-life-ish (1/s)
  /** Full-width drag sweeps ~70 degrees of yaw regardless of DPI. */
  private static readonly DRAG_SCALE = degToRad(70);

  constructor(private el: HTMLElement) {
    el.classList.add("grabbable");
    el.addEventListener("pointerdown", this.onDown);
    el.addEventListener("pointermove", this.onMove);
    el.addEventListener("pointerup", this.onUp);
    el.addEventListener("pointercancel", this.onUp);
  }

  private onDown = (e: PointerEvent): void => {
    this.dragging = true;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    this.lastMoveTime = e.timeStamp;
    this.velYaw = 0;
    this.velPitch = 0;
    this.el.setPointerCapture(e.pointerId);
    this.el.classList.add("grabbing");
  };

  private onMove = (e: PointerEvent): void => {
    if (!this.dragging) return;
    // Client (CSS-pixel) coordinates: identical feel on high-DPI displays.
    const w = this.el.clientWidth || 1;
    const dYaw = ((e.clientX - this.lastX) / w) * OrbitCamera.DRAG_SCALE;
    const dPitch = ((e.clientY - this.lastY) / w) * OrbitCamera.DRAG_SCALE;
    this.lastX = e.clientX;
    this.lastY = e.clientY;

    // Drag right -> look left (grab-the-world convention); drag up -> look up.
    this.targetYaw -= dYaw;
    this.targetPitch = clamp(
      this.targetPitch + dPitch,
      OrbitCamera.PITCH_MIN,
      OrbitCamera.PITCH_MAX,
    );

    const dt = Math.max((e.timeStamp - this.lastMoveTime) / 1000, 1e-3);
    this.lastMoveTime = e.timeStamp;
    this.velYaw = -dYaw / dt;
    this.velPitch = dPitch / dt;
  };

  private onUp = (e: PointerEvent): void => {
    this.dragging = false;
    this.el.releasePointerCapture(e.pointerId);
    this.el.classList.remove("grabbing");
  };

  /** Advance the spring/inertia simulation by dt seconds. */
  update(dt: number): void {
    if (!this.dragging) {
      // Inertia: released velocity keeps nudging the target, decaying away.
      const decay = Math.exp(-OrbitCamera.INERTIA_DECAY * dt);
      this.targetYaw += this.velYaw * dt;
      this.targetPitch = clamp(
        this.targetPitch + this.velPitch * dt,
        OrbitCamera.PITCH_MIN,
        OrbitCamera.PITCH_MAX,
      );
      this.velYaw *= decay;
      this.velPitch *= decay;
    }
    // Exponential smoothing toward the target (frame-rate independent).
    const a = 1 - Math.exp(-OrbitCamera.SMOOTHING * dt);
    this.yaw += (this.targetYaw - this.yaw) * a;
    this.pitch += (this.targetPitch - this.pitch) * a;
  }
}

function degToRad(d: number): number {
  return (d * Math.PI) / 180;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(Math.max(v, lo), hi);
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("Failed to allocate shader object");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile error:\n${log ?? "(no log)"}`);
  }
  return shader;
}

function createProgram(gl: WebGL2RenderingContext, vsSource: string, fsSource: string): WebGLProgram {
  const program = gl.createProgram();
  if (!program) throw new Error("Failed to allocate program object");
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSource);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSource);
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  // Shaders can be flagged for deletion once linked.
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Program link error:\n${log ?? "(no log)"}`);
  }
  return program;
}

function main(): void {
  const canvas = document.getElementById("sky") as HTMLCanvasElement | null;
  if (!canvas) throw new Error("Canvas element #sky not found");
  canvas.classList.add("sky-canvas");

  const gl = canvas.getContext("webgl2", {
    antialias: false, // fragment shader supplies its own dithering; MSAA is wasted on a fullscreen quad
    depth: false,
    stencil: false,
    alpha: false,
    powerPreference: "high-performance",
  });
  if (!gl) throw new Error("WebGL2 is not supported by this browser");

  const program = createProgram(gl, vertexSource, fragmentSource);
  gl.useProgram(program);

  const uResolution = gl.getUniformLocation(program, "uResolution");
  const uTime = gl.getUniformLocation(program, "uTime");
  const uCamYaw = gl.getUniformLocation(program, "uCamYaw");
  const uCamPitch = gl.getUniformLocation(program, "uCamPitch");

  const camera = new OrbitCamera(canvas);

  // WebGL2 requires a bound VAO even when attributes are unused.
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  function resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_DEVICE_PIXEL_RATIO);
    const width = Math.round(canvas!.clientWidth * dpr);
    const height = Math.round(canvas!.clientHeight * dpr);
    if (canvas!.width !== width || canvas!.height !== height) {
      canvas!.width = width;
      canvas!.height = height;
      gl!.viewport(0, 0, width, height);
    }
  }

  const start = performance.now();
  let lastFrame = start;
  function frame(now: number): void {
    const dt = Math.min((now - lastFrame) / 1000, 0.1);
    lastFrame = now;
    camera.update(dt);

    resize();
    gl!.uniform2f(uResolution, canvas!.width, canvas!.height);
    gl!.uniform1f(uTime, (now - start) / 1000);
    gl!.uniform1f(uCamYaw, camera.yaw);
    gl!.uniform1f(uCamPitch, camera.pitch);
    gl!.drawArrays(gl!.TRIANGLES, 0, 3);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// Vercel Web Analytics. This vanilla Vite app uses the framework-agnostic
// inject(); it is a no-op in local dev and only reports once deployed.
inject();

main();
