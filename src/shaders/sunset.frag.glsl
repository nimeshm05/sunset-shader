#version 300 es
/*
 * ============================================================================
 *  CINEMATIC SUNSET — single-pass procedural sky simulation
 * ============================================================================
 *
 *  A fully analytical WebGL2 / GLSL ES 3.00 fragment shader. No textures,
 *  no lookup tables, no external assets. The image is built in stages:
 *
 *    1. Noise foundation      — hash, 3D value noise, rotated fbm, warping
 *    2. Atmospheric model     — Rayleigh + Mie + ozone single scattering
 *    3. Sun radiance          — limb-darkened HDR disc + Mie aureole
 *    4. Clouds                — two analytic layers (stratocumulus + cirrus)
 *                               with self-shadowing, rim light, forward
 *                               scattering and aerial perspective
 *    4b. Ocean                — raymarched Gerstner-wave height field with
 *                               analytic normals, Fresnel sky reflection,
 *                               microfacet sun glints, absorption + SSS,
 *                               all lit by the same atmospheric model
 *    5. Camera / post         — analytic glare (bloom), exposure adaptation,
 *                               ACES filmic tone map, color grade, vignette,
 *                               chromatic aberration, dithering
 *
 *  All radiometric quantities before tone mapping are HDR (unbounded).
 *  Units are loosely physical: optical depths are real zenith values, the
 *  sun intensity is an arbitrary HDR scale chosen for the tone mapper.
 * ============================================================================
 */
precision highp float;

uniform vec2  uResolution;   // framebuffer size in pixels
uniform float uTime;         // seconds since load
uniform float uCamYaw;       // damped orbit camera yaw (radians)
uniform float uCamPitch;     // damped orbit camera pitch (radians)

out vec4 fragColor;

/* ============================================================================
 *  CONSTANTS
 * ==========================================================================*/

const float PI  = 3.141592653589793;
const float TAU = 6.283185307179586;

/* --- Atmosphere: zenith optical depths per RGB channel (~680/550/440 nm) ---
 * Rayleigh: beta(lambda) ~ 1/lambda^4 integrated over an 8 km scale height.
 * These are close to measured values for a clear standard atmosphere.      */
const vec3 TAU_RAYLEIGH = vec3(0.0464, 0.1086, 0.2650);

/* Mie (aerosol) zenith optical depth. Nearly wavelength-neutral, with a
 * slight warm bias typical of larger haze particles. Scaled at runtime by
 * a slowly drifting "turbidity" factor so the mood evolves over minutes.  */
const vec3 TAU_MIE_BASE = vec3(0.0480, 0.0440, 0.0400);

/* Ozone (Chappuis band) absorbs green/orange far more than blue. This term
 * is what physically produces the violet-magenta twilight afterglow: it
 * removes green from long horizontal paths, leaving red + blue = magenta. */
const vec3 TAU_OZONE = vec3(0.0210, 0.0600, 0.0035);

const float MIE_G_SKY   = 0.76;  // aerosol forward-scattering anisotropy
const float MIE_G_CLOUD = 0.55;  // droplet forward-scattering (silver lining)

const float SUN_INTENSITY  = 22.0;   // HDR scale of in-scattered sunlight
const float SUN_DISC_I     = 42.0;   // HDR radiance of the solar disc
const float SUN_ANG_RADIUS = 0.0330; // ~1.9 deg: enlarged for cinematic scale

/* --- Scene geometry (meters) --- */
const float CAM_HEIGHT    = 8.0;     // camera altitude above sea level
const float STRATO_HEIGHT = 1500.0;  // low cloud deck
const float CIRRUS_HEIGHT = 6500.0;  // high ice wisps

/* --- Animation time scales: everything is *very* slow --- */
const float CLOUD_EVOLVE_RATE = 0.010;  // noise 3rd-dimension drift (units/s)
const float WARP_EVOLVE_RATE  = 0.006;
const float SUN_DRIFT_PERIOD  = 540.0;  // sun elevation cycle (9 minutes)
const float HAZE_DRIFT_PERIOD = 380.0;  // turbidity / mood cycle

/* Octave-to-octave rotation for fbm. An orthonormal-ish matrix that both
 * rotates and mixes axes, destroying the lattice alignment that causes
 * visible tiling and axis-parallel artifacts in naive fbm.               */
const mat3 FBM_ROT = mat3( 0.00,  0.80,  0.60,
                          -0.80,  0.36, -0.48,
                          -0.60, -0.48,  0.64);

/* ============================================================================
 *  1. NOISE FOUNDATION
 * ==========================================================================*/

/* Sine-free hash (D. Hoskins). Avoids the precision blow-up of the classic
 * fract(sin(dot(...))*43758.5453) hash on large coordinates.              */
float hash13(vec3 p) {
    p  = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

/* Screen-space white hash for dithering / CA jitter. */
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

/* 3D value noise with quintic (C2) interpolation — no derivative kinks,
 * which matters because clouds are shaded from density *differences*.    */
float noise3(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    float n000 = hash13(i + vec3(0.0, 0.0, 0.0));
    float n100 = hash13(i + vec3(1.0, 0.0, 0.0));
    float n010 = hash13(i + vec3(0.0, 1.0, 0.0));
    float n110 = hash13(i + vec3(1.0, 1.0, 0.0));
    float n001 = hash13(i + vec3(0.0, 0.0, 1.0));
    float n101 = hash13(i + vec3(1.0, 0.0, 1.0));
    float n011 = hash13(i + vec3(0.0, 1.0, 1.0));
    float n111 = hash13(i + vec3(1.0, 1.0, 1.0));

    return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
               mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}

/* Fractal Brownian motion, 5 octaves. Each octave is rotated by FBM_ROT and
 * scaled by a non-integer lacunarity (2.02) so octave lattices never align:
 * this removes both visible tiling and the "waffle" artifact of value noise. */
float fbm5(vec3 p) {
    float amp = 0.5;
    float sum = 0.0;
    for (int i = 0; i < 5; i++) {
        sum += amp * noise3(p);
        p    = FBM_ROT * p * 2.02;
        amp *= 0.5;
    }
    return sum; // ~[0.03 .. 0.97], mean ~0.48
}

/* Cheaper 3-octave variant for secondary lookups (shadow probes, wisps). */
float fbm3(vec3 p) {
    float amp = 0.5;
    float sum = 0.0;
    for (int i = 0; i < 3; i++) {
        sum += amp * noise3(p);
        p    = FBM_ROT * p * 2.02;
        amp *= 0.5;
    }
    return sum;
}

/* Domain warp field: two decorrelated fbm channels form a 2D displacement.
 * Warping the cloud domain by another noise field breaks up the statistical
 * self-similarity of raw fbm — the single most effective trick against
 * "obviously procedural" cloudscapes.                                     */
vec2 warpField(vec3 q) {
    return vec2(fbm3(q + vec3( 0.0, 0.0,  3.17)),
                fbm3(q + vec3(5.20, 1.3, -7.70))) - 0.5;
}

/* ============================================================================
 *  2. ATMOSPHERIC SCATTERING
 * ==========================================================================*/

/* Relative optical air mass for a ray leaving the ground at a given
 * elevation (Kasten & Young 1989). ~1 at zenith, ~38 at the horizon.
 * This single analytic function replaces a full raymarched atmosphere:
 * optical depth along any ray = zenith optical depth * airmass.          */
float airmass(float cosElev) {
    float c = clamp(cosElev, -0.035, 1.0);
    float zenithDeg = degrees(acos(clamp(c, 0.0, 1.0)));
    float m = 1.0 / (max(c, 0.0) + 0.50572 * pow(96.07995 - zenithDeg, -1.6364));
    return min(m, 42.0);
}

/* Rayleigh phase function (normalized). */
float phaseRayleigh(float mu) {
    return 3.0 / (16.0 * PI) * (1.0 + mu * mu);
}

/* Henyey–Greenstein phase function (normalized) for aerosol / droplet
 * forward scattering.                                                     */
float phaseHG(float mu, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * mu, 1.5));
}

/* Turbidity drifts slowly (period ~6 min) so haze density — and with it the
 * warmth of the horizon — shifts imperceptibly over several minutes.      */
vec3 tauMie() {
    float turbidity = 1.0 + 0.30 * sin(uTime * TAU / HAZE_DRIFT_PERIOD + 1.7);
    return TAU_MIE_BASE * turbidity;
}

/* Full-column transmittance toward a direction with the given elevation
 * cosine. Ozone lives high in the stratosphere, so its slant path saturates
 * near the horizon — modeled by clamping its effective airmass.           */
vec3 atmosphereTransmittance(float cosElev) {
    float m = airmass(cosElev);
    vec3 tau = (TAU_RAYLEIGH + tauMie()) * m + TAU_OZONE * min(m, 12.0);
    return exp(-tau);
}

/* Single-scattering sky radiance.
 *
 * The classic Hoffman/Preetham decomposition:
 *     L = E_sun * (beta_R * phase_R + beta_M * phase_M) / (beta_R + beta_M)
 *              * T_sun * (1 - exp(-tau_view))
 *
 * with one physically-motivated refinement: Rayleigh scattering happens
 * high in the atmosphere (light reaching those molecules is only mildly
 * reddened), while Mie scattering happens in the low haze layer (light
 * reaching it near sunset is strongly reddened). We therefore evaluate the
 * sun transmittance at two different effective air masses. This is what
 * keeps the zenith deep cobalt while the horizon near the sun turns gold —
 * a single shared transmittance would muddy both.                         */
vec3 skyRadiance(vec3 rd, vec3 sunDir) {
    vec3 tauM = tauMie();

    float mView = airmass(rd.y);
    vec3 tauView = (TAU_RAYLEIGH + tauM) * mView + TAU_OZONE * min(mView, 12.0);
    vec3 extView = exp(-tauView);          // view-path transmittance
    vec3 scatterAmt = 1.0 - extView;       // fraction of light scattered in

    float mSun = airmass(sunDir.y);
    vec3 tauZen = TAU_RAYLEIGH + tauM + TAU_OZONE;

    /* Where along the ray does the scattering happen? Zenith-facing rays
     * collect light scattered high up (mildly reddened sun); horizon-grazing
     * rays collect light scattered low down, where the sun has already been
     * dragged through a huge slant path and arrives deep orange. Blending
     * the effective sun transmittance on view airmass reproduces the long,
     * warm golden band that hugs the whole horizon at sunset.             */
    float horizonness = 1.0 - exp(-(mView - 1.0) * 0.11);
    vec3 tSunRay = exp(-tauZen * mSun * mix(0.18, 0.62, horizonness));
    vec3 tSunLow = exp(-tauZen * mSun * 1.15);   // illumination of low haze

    float mu = dot(rd, sunDir);
    float phR = phaseRayleigh(mu);
    float phM = phaseHG(mu, MIE_G_SKY);

    /* Energy-weighted blend of the two scattering species. */
    vec3 betaSum = TAU_RAYLEIGH + tauM;
    vec3 inscatter = SUN_INTENSITY *
        (TAU_RAYLEIGH * phR * tSunRay + tauM * phM * tSunLow * 1.6) / betaSum *
        scatterAmt;

    /* Cheap multiple-scattering floor: a dim, cool ambient proportional to
     * how much atmosphere the ray traverses. Prevents an unnaturally black
     * zenith, keeps it convincingly cobalt, and fills the shadowed side of
     * the sky dome. Deliberately blue: multiply-scattered twilight light
     * has bounced through the ozone layer and upper (Rayleigh) atmosphere. */
    vec3 multiScatter = vec3(0.013, 0.033, 0.105) * scatterAmt.b;

    return inscatter + multiScatter;
}

/* ============================================================================
 *  3. SUN RADIANCE
 * ==========================================================================*/

/* Limb-darkened solar disc. The disc radiance is attenuated by the full
 * atmospheric column, so at 2 degrees elevation it arrives deep orange-red
 * "for free" — no artistic tinting needed.                                */
vec3 sunRadiance(vec3 rd, vec3 sunDir, vec3 sunTransmit) {
    float mu = dot(rd, sunDir);
    float angle = acos(clamp(mu, -1.0, 1.0));

    /* Soft analytic edge: ~0.05 deg of penumbra hides pixel stairstepping. */
    float disc = smoothstep(SUN_ANG_RADIUS, SUN_ANG_RADIUS - 0.0009, angle);

    /* Empirical limb darkening: I(r) = 1 - u * (1 - sqrt(1 - r^2)).       */
    float r = clamp(angle / SUN_ANG_RADIUS, 0.0, 1.0);
    float limbMu = sqrt(max(1.0 - r * r, 0.0));
    float limb = mix(1.0 - 0.62, 1.0, limbMu);

    return SUN_DISC_I * sunTransmit * disc * limb;
}

/* ============================================================================
 *  4. CLOUDS
 * ==========================================================================*/

/* --- Low stratocumulus deck ------------------------------------------------
 * Density is evaluated on the ray/plane intersection with the layer.
 * The domain is (x, z, evolutionTime): time enters as the *third noise
 * dimension*, so clouds morph in place instead of sliding across the sky.
 *
 * Structure  = low-frequency warped fbm masses
 *            + higher-frequency wisps riding on the same warp
 * Coverage   = remap threshold, biased denser toward the horizon.         */
float stratoDensity(vec3 q, float coverage, out vec3 qWarped) {
    vec2 w = warpField(q * 0.55);
    qWarped = q + vec3(w * 1.9, 0.0);

    float masses = fbm5(qWarped);                          // broad shapes
    float wisps  = fbm3(qWarped * 3.7 + vec3(11.0, 7.0, 0.0)); // fine detail
    float d = masses + 0.38 * (wisps - 0.5);

    return smoothstep(coverage, coverage + 0.30, d);
}

/* Cheap density probe toward the sun, reusing the already-computed warp.
 * Used for self-shadowing and rim lighting; the warp field varies slowly
 * enough that freezing it over the probe distance is visually lossless.  */
float stratoDensityProbe(vec3 qWarped, float coverage) {
    float masses = fbm3(qWarped);
    return smoothstep(coverage, coverage + 0.30, masses);
}

/* --- High cirrus -----------------------------------------------------------
 * Thin ice-crystal streaks: strongly anisotropic domain (stretched along a
 * "wind" axis), higher frequency, sharpened remap, low opacity.           */
float cirrusDensity(vec3 q) {
    vec2 w = warpField(q * 0.8 + vec3(9.0, 3.0, 0.0));
    vec3 qw = q + vec3(w * 1.1, 0.0);
    float d = fbm5(vec3(qw.x * 0.45, qw.y * 2.1, qw.z));
    /* Sharpen and thin out: cirrus is sparse and semi-transparent. */
    return smoothstep(0.52, 0.80, d) * 0.55;
}

/* Shade a cloud sample and composite it over the background.
 *
 *   density      cloud opacity driver at this pixel
 *   shadow       0 = fully self-shadowed interior, 1 = directly sunlit
 *   rim          sun-facing density gradient (edge highlight)
 *   mu           cos(angle between view ray and sun)
 *   dist         distance to the cloud sample (for aerial perspective)
 *   skyBehind    background radiance being occluded                       */
vec3 shadeCloud(vec3 skyBehind, float density, float shadow, float rim,
                float mu, float dist, vec3 sunWarm, vec3 skyCool,
                float maxAlpha) {
    /* Forward scattering through the slab: clouds between us and the sun
     * glow at their thin parts (silver lining).                           */
    float forward = phaseHG(mu, MIE_G_CLOUD) * 4.0 * PI;   // ~1 at 90 deg
    float silver  = forward * mix(1.0, 0.25, density);      // thin => bright

    /* Direct warm sunlight, gated by self-shadowing, plus rim highlight.  */
    vec3 direct = sunWarm * (0.18 + 0.82 * shadow) * (0.55 + 0.45 * silver)
                + sunWarm * rim * 1.6;

    /* Shadowed interiors fall toward the cool ambient of the sky dome,
     * shifted violet — shadowed cloud material is lit by the blue sky,
     * and ozone absorption tilts that ambient toward violet at sunset.    */
    vec3 ambient = skyCool * (0.65 + 0.35 * (1.0 - density));
    vec3 violetShift = vec3(0.42, 0.33, 0.58);
    ambient = mix(ambient, ambient.g * 2.2 * violetShift, (1.0 - shadow) * 0.45);

    vec3 cloudCol = direct + ambient;

    /* Aerial perspective: extinction + in-scatter toward the sky color.
     * Distant cloud decks dissolve into the haze instead of staying crisp. */
    float aerial = exp(-dist * 3.0e-5);
    cloudCol = mix(skyBehind, cloudCol, aerial);

    float alpha = clamp(density * maxAlpha, 0.0, 1.0) * aerial;
    return mix(skyBehind, cloudCol, alpha);
}

/* Render both cloud layers over the sky. Layers are composited far-to-near
 * (cirrus first, stratocumulus on top). Returns radiance + total opacity
 * near the sun so the post-process glare can be partially occluded.       */
vec3 renderClouds(vec3 sky, vec3 rd, vec3 sunDir, vec3 sunTransmitLow,
                  out float cloudCover) {
    cloudCover = 0.0;

    /* Clouds exist only above the horizon; fade smoothly at grazing angles
     * (mask instead of branch).                                           */
    float upMask = smoothstep(0.015, 0.09, rd.y);
    float rdy = max(rd.y, 0.015);

    /* Illumination colors shared by both layers. */
    vec3 sunWarm = SUN_INTENSITY * 0.62 * sunTransmitLow;      // warm gold key
    vec3 skyCool = skyRadiance(normalize(vec3(rd.x, 1.0, rd.z)), sunDir) * 0.9
                 + vec3(0.012, 0.016, 0.038);                  // cool fill
    float mu = dot(rd, sunDir);
    float tEvolve = uTime * CLOUD_EVOLVE_RATE;

    vec3 col = sky;

    /* ---- Cirrus (far layer) ---- */
    {
        float tHit = (CIRRUS_HEIGHT - CAM_HEIGHT) / rdy;
        vec3 hit = rd * tHit;
        vec3 q = vec3(hit.xz * 1.35e-4, tEvolve * 0.7);
        float d = cirrusDensity(q) * upMask;

        /* Cirrus is optically thin: skip self-shadowing, keep strong
         * forward scattering so streaks near the sun glow.                */
        col = shadeCloud(col, d, 0.85, 0.0, mu, tHit,
                         sunWarm, skyCool, 0.55);
        cloudCover += d * 0.35;
    }

    /* ---- Stratocumulus (near layer) ---- */
    {
        float tHit = (STRATO_HEIGHT - CAM_HEIGHT) / rdy;
        vec3 hit = rd * tHit;
        vec3 q = vec3(hit.xz * 5.0e-4, tEvolve);

        /* Coverage grows toward the horizon: at grazing angles the eye
         * looks through more of the deck, so apparent density rises.      */
        float coverage = mix(0.35, 0.56, smoothstep(0.02, 0.45, rd.y));

        vec3 qWarped;
        float d = stratoDensity(q, coverage, qWarped) * upMask;

        /* Self-shadowing: probe density a short step toward the sun in the
         * warped domain. High probe density = deep interior = shadowed.
         * The probe also feeds rim lighting: where density falls off toward
         * the sun, the edge catches direct warm light.                    */
        vec3 sunStep = vec3(sunDir.x, sunDir.z, 0.0) * 0.35
                     + vec3(0.0, 0.0, 0.02);
        float dSun = stratoDensityProbe(qWarped + sunStep, coverage - 0.04);
        float shadow = exp(-dSun * 2.6);
        float rim = clamp(d - dSun, 0.0, 1.0) * smoothstep(0.0, 0.35, d);

        col = shadeCloud(col, d, shadow, rim, mu, tHit,
                         sunWarm, skyCool, 0.92);
        cloudCover += d;
    }

    cloudCover = clamp(cloudCover, 0.0, 1.0);
    return col;
}

/* ============================================================================
 *  4b. OCEAN
 *
 *  A calm open-ocean surface, raymarched against an analytically displaced
 *  height field built from superimposed Gerstner-style waves. Everything the
 *  water shows — reflections, glints, haze — is sampled from the same
 *  atmospheric model that renders the sky, so sky and sea form one coherent
 *  lighting system.
 * ==========================================================================*/

const float GRAVITY   = 9.81;
const int   NUM_WAVES = 10;
const float OCEAN_T_MAX = 20000.0;  // beyond this, haze has taken over anyway

/* Wave spectrum: (dir.x, dir.z, wavelength m, amplitude m).
 * Three bands — swells, medium waves, short chop — fanned around a common
 * wind heading with irregular angular spacing so no two components align.
 * Each wave gets the physically correct deep-water dispersion speed
 * (omega = sqrt(g * k)), so long swells outrun short chop naturally.       */
const vec4 OCEAN_WAVES[NUM_WAVES] = vec4[](
    /* --- large swells: gentle, long-period rollers --- */
    vec4(-0.801,  0.599, 95.0, 0.550),
    vec4(-0.988,  0.161, 67.0, 0.400),
    vec4(-0.538,  0.843, 51.0, 0.300),
    /* --- medium wind waves --- */
    vec4(-0.932,  0.362, 23.0, 0.140),
    vec4(-0.688,  0.726, 16.0, 0.100),
    vec4(-1.000,  0.002, 11.0, 0.075),
    vec4(-0.470,  0.883,  7.5, 0.050),
    /* --- short chop --- */
    vec4(-0.866,  0.499,  3.8, 0.032),
    vec4(-0.988, -0.157,  2.6, 0.022),
    vec4(-0.370,  0.929,  1.7, 0.016));

/* Global slow-motion factor: physical dispersion but at a meditative pace. */
const float WAVE_TIME_SCALE = 0.6;

/* Peaked (non-sinusoidal) wave profile: exp(sin - 1) has sharp crests and
 * long flat troughs, the characteristic asymmetry of real water that a raw
 * sine sum never shows. The -0.45 recenters the profile around sea level.  */

/* Wave-group envelope: a very-low-frequency noise (150 m+ features) slowly
 * swelling and fading whole wave groups over time. This is what makes the
 * surface evolve organically rather than translate — the third noise
 * dimension is time, exactly like the clouds.                              */
float waveEnvelope(vec2 xz) {
    return 0.55 + 0.90 * fbm3(vec3(xz * 0.006, uTime * 0.010));
}

/* Height only — used inside the intersection loop, so it must stay cheap:
 * pure sin/exp per wave, no noise. `dist` drives the wavelength-aware LOD:
 * waves shorter than the pixel footprint fade out analytically, which is
 * simultaneously anti-aliasing and an optimization.                        */
float oceanHeight(vec2 xz, float dist, float env) {
    float t = uTime * WAVE_TIME_SCALE;
    float h = 0.0;
    for (int i = 0; i < NUM_WAVES; i++) {
        vec4 w = OCEAN_WAVES[i];
        float k = TAU / w.z;
        float ph = k * dot(w.xy, xz) + sqrt(GRAVITY * k) * t;
        float lod = exp(-dist * 0.045 / w.z);
        h += w.w * (exp(sin(ph) - 1.0) - 0.45) * lod;
    }
    return h * env;
}

/* Full surface evaluation: accumulates the analytic gradient (dh/dx, dh/dz)
 * in the same pass — exact normals from wave derivatives, no finite
 * differences. Fine capillary ripples are added to the *normal only* (their
 * height is sub-centimeter and invisible; their slope is what sparkles).   */
vec3 oceanNormal(vec2 xz, float dist, float env) {
    float t = uTime * WAVE_TIME_SCALE;
    vec2 grad = vec2(0.0);
    for (int i = 0; i < NUM_WAVES; i++) {
        vec4 w = OCEAN_WAVES[i];
        float k = TAU / w.z;
        float ph = k * dot(w.xy, xz) + sqrt(GRAVITY * k) * t;
        float lod = exp(-dist * 0.045 / w.z);
        /* d/dx [A * exp(sin(ph) - 1)] = A * exp(sin(ph)-1) * cos(ph) * k * dir */
        grad += w.w * exp(sin(ph) - 1.0) * cos(ph) * k * w.xy * lod;
    }
    /* The envelope's own spatial gradient is negligible (150 m features vs
     * meter-scale waves), so scaling the wave gradient by it is accurate.  */
    grad *= env;

    /* Capillary detail: gradient of a small warped noise field, evolving
     * through its time dimension. Faded aggressively with distance — beyond
     * ~150 m these ripples are far below one pixel.                        */
    float capFade = exp(-dist * 0.022);
    vec2 capGrad = vec2(0.0);
    if (capFade > 0.004) {
        vec3 cq = vec3(xz * 1.15, uTime * 0.22);
        float e = 0.28;
        float c0 = fbm3(cq);
        capGrad = vec2(fbm3(cq + vec3(e, 0.0, 0.0)) - c0,
                       fbm3(cq + vec3(0.0, e, 0.0)) - c0) / e;
        grad += capGrad * 0.075 * capFade;
    }

    return normalize(vec3(-grad.x, 1.0, -grad.y));
}

/* Ray / height-field intersection. The sea is calm (crests < ~1.5 m) and the
 * camera is above every crest, so a relaxed Newton iteration along the ray
 * converges in a handful of steps: start at the plane through the highest
 * possible crest, then repeatedly step forward by the current height error
 * projected onto the ray. Fixed iteration count keeps it branch-free.      */
float traceOcean(vec3 ro, vec3 rd, float env) {
    float denom = max(-rd.y, 1.5e-3);
    float t = max((ro.y - 1.6) / denom, 0.0);
    for (int i = 0; i < 8; i++) {
        vec3 p = ro + rd * t;
        float h = oceanHeight(p.xz, t, env);
        /* Small slope guard in the denominator keeps grazing rays stable. */
        t += (p.y - h) / (denom + 0.02) * 0.9;
        t = min(t, OCEAN_T_MAX);
    }
    return t;
}

/* Sky seen along a reflected ray, including a soft imprint of the cloud
 * deck. Full cloud rendering along every reflected ray would double the
 * frame cost; instead one cheap warped-fbm deck — sampled in the *same*
 * domain as the real stratocumulus so shapes correspond — is composited
 * with simplified shading. Water reflections are dimmed by Fresnel and
 * blurred by wave normals anyway, so the approximation is invisible.       */
vec3 reflectedSky(vec3 rd, vec3 sunDir, vec3 sunWarm) {
    vec3 col = skyRadiance(rd, sunDir);

    float rdy = max(rd.y, 0.03);
    float tHit = STRATO_HEIGHT / rdy;
    vec3 q = vec3(rd.xz / rdy * STRATO_HEIGHT * 5.0e-4,
                  uTime * CLOUD_EVOLVE_RATE);
    float coverage = mix(0.35, 0.56, smoothstep(0.02, 0.45, rd.y));
    float d = smoothstep(coverage, coverage + 0.32, fbm5(q));

    float aerial = exp(-tHit * 3.0e-5);
    vec3 cloudCol = sunWarm * 0.55 + col * 0.6;
    return mix(col, cloudCol, d * 0.85 * aerial);
}

/* Shade the traced water surface.
 *
 * The BRDF splits into:
 *   - Fresnel-weighted mirror term: the reflected sky/cloud radiance
 *   - microfacet sun glint lane (Blinn lobe, roughness grows with distance
 *     as more wave facets crowd into each pixel)
 *   - transmitted body color: absorption-tinted upwelling light plus a
 *     subsurface forward-scattering glow on sunward wave flanks
 * followed by aerial perspective toward the horizon sky.                   */
vec3 shadeOcean(vec3 ro, vec3 rd, vec3 sunDir, vec3 sunTransmit,
                vec3 sunTransmitLow) {
    /* The group envelope varies over ~150 m, while the intersection refines
     * the hit by meters — so evaluating it once at the flat-plane hit is
     * accurate and keeps noise out of the trace loop.                      */
    float tPlane = min(ro.y / max(-rd.y, 1.5e-3), OCEAN_T_MAX);
    float env = waveEnvelope((ro + rd * tPlane).xz);

    float t = traceOcean(ro, rd, env);
    vec3 p = ro + rd * t;
    vec3 N = oceanNormal(p.xz, t, env);

    vec3 V = -rd;
    float cosV = clamp(dot(N, V), 0.0, 1.0);

    /* Schlick Fresnel, F0 = 0.02 (water/air): reflectance climbs steeply
     * toward grazing incidence, so the far water becomes pure sky mirror.  */
    float fresnel = 0.02 + 0.98 * pow(1.0 - cosV, 5.0);

    vec3 sunWarm = SUN_INTENSITY * 0.62 * sunTransmitLow;

    /* --- Mirror term: reflected atmosphere + soft clouds --- */
    vec3 R = reflect(rd, N);
    R.y = max(R.y, 0.015);            // the surface only reflects the sky
    R = normalize(R);
    vec3 refl = reflectedSky(R, sunDir, sunWarm);

    /* --- Sun glint lane ---
     * Normalized Blinn microfacet lobe. Near the camera individual facets
     * resolve (high exponent -> hard sparkle); with distance the effective
     * roughness rises (many facets per pixel) and the lane widens into the
     * long shimmering path. Because N comes from the real wave geometry,
     * the lane meanders and breaks up instead of forming a straight stripe. */
    vec3 H = normalize(V + sunDir);
    float ndh = clamp(dot(N, H), 0.0, 1.0);
    float nearness = exp(-t * 6.0e-4);
    float specPow = mix(42.0, 680.0, nearness);
    float spec = pow(ndh, specPow) * (specPow + 8.0) / (8.0 * PI);
    float fresnelH = 0.02 + 0.98 * pow(1.0 - clamp(dot(H, V), 0.0, 1.0), 5.0);
    vec3 glint = SUN_DISC_I * sunTransmit * spec * fresnelH * 0.55;

    /* --- Transmitted body color ---
     * Water absorbs red within meters, leaving a dark blue-green upwelling
     * lit by the sky dome. At sunset that ambient is dim: deep water reads
     * nearly black with a cool cast, exactly as it should.                 */
    vec3 skyAmb = skyRadiance(vec3(0.0, 1.0, 0.0), sunDir);
    vec3 body = vec3(0.055, 0.155, 0.210) * skyAmb * 1.5;

    /* Subsurface scattering approximation: looking toward the sun through
     * wave crests, some light transmits through the thin water and scatters
     * back out cyan-green. Gated on view-sun alignment and crest height.   */
    float crest = clamp(p.y * 1.4 + 0.35, 0.0, 1.0);
    float sss = pow(clamp(dot(rd, sunDir), 0.0, 1.0), 4.0) * crest;
    body += vec3(0.10, 0.26, 0.23) * sunTransmitLow * SUN_INTENSITY * 0.085 * sss;

    vec3 col = mix(body, refl, fresnel) + glint;

    /* --- Aerial perspective ---
     * The same haze that softens distant clouds swallows the far water and
     * welds it to the sky: no hard horizon line can survive this blend.    */
    vec3 horizonSky = skyRadiance(normalize(vec3(rd.x, 0.0015, rd.z)), sunDir);
    float haze = exp(-t * 3.5e-4);
    return mix(horizonSky, col, haze);
}

/* ============================================================================
 *  5. CAMERA / POST-PROCESSING
 * ==========================================================================*/

/* Analytic bloom. A real bloom pass blurs bright pixels; in a single pass we
 * instead add the camera's glare point-spread-function around the one known
 * HDR source — the sun. Two Lorentzian lobes approximate the measured PSF of
 * photographic lenses: a tight bright halo plus a very wide dim veil.
 * `occlusion` lets foreground clouds damp the glare they block.           */
vec3 bloomApprox(vec3 rd, vec3 sunDir, vec3 sunTint, float occlusion) {
    float angle = acos(clamp(dot(rd, sunDir), -1.0, 1.0));
    float th = max(angle, SUN_ANG_RADIUS * 0.75);

    float halo = 0.030 / (1.0 + pow(th / 0.040, 2.0));  // tight warm halo
    float veil = 0.008 / (1.0 + pow(th / 0.400, 2.0));  // wide soft veil

    return SUN_DISC_I * sunTint * (halo + veil) * occlusion;
}

/* ACES filmic tone map — Stephen Hill's fitted RRT+ODT approximation.
 * Matrices are in GLSL column-major layout.                               */
const mat3 ACES_INPUT = mat3(
    0.59719, 0.07600, 0.02840,
    0.35458, 0.90834, 0.13383,
    0.04823, 0.01566, 0.83777);

const mat3 ACES_OUTPUT = mat3(
     1.60475, -0.10208, -0.00327,
    -0.53108,  1.10813, -0.07276,
    -0.07367, -0.00605,  1.07602);

vec3 rrtOdtFit(vec3 v) {
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

vec3 acesToneMap(vec3 hdr) {
    vec3 c = ACES_INPUT * hdr;
    c = rrtOdtFit(c);
    c = ACES_OUTPUT * c;
    return clamp(c, 0.0, 1.0);
}

/* Gentle photographic grade, applied after tone mapping.
 * Shadows lift faintly toward teal, highlights stay warm — the classic
 * cinematic complement — plus a mild saturation touch. Kept subtle so the
 * physical palette of the atmosphere still reads as real.                 */
vec3 colorGrade(vec3 c) {
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));

    vec3 shadowTint    = vec3(0.92, 1.015, 1.06);
    vec3 highlightTint = vec3(1.045, 1.0, 0.94);
    vec3 tint = mix(shadowTint, highlightTint, smoothstep(0.10, 0.75, luma));
    c *= tint;

    c = mix(vec3(luma), c, 1.12);          // +12% saturation
    c = c * 1.01 - 0.004;                  // hair of contrast, floor the blacks
    return clamp(c, 0.0, 1.0);
}

/* ============================================================================
 *  SCENE ASSEMBLY
 * ==========================================================================*/

/* Sun direction drifts imperceptibly: elevation breathes between ~1.6 and
 * ~3.4 degrees over nine minutes, azimuth wanders a fraction of a degree.
 * Over several minutes the whole palette warms and cools in response.     */
vec3 sunDirection() {
    float elev = radians(2.0 + 0.8 * sin(uTime * TAU / SUN_DRIFT_PERIOD));
    float azim = radians(-4.0 + 0.6 * sin(uTime * TAU / (SUN_DRIFT_PERIOD * 1.7)));
    float ce = cos(elev);
    return normalize(vec3(sin(azim) * ce, sin(elev), cos(azim) * ce));
}

/* Full HDR scene radiance for one view ray. */
vec3 renderScene(vec3 rd, vec3 sunDir) {
    vec3 sunTransmit = atmosphereTransmittance(sunDir.y);
    /* Reddened low-atmosphere sun light used for cloud illumination. */
    vec3 tauZen = TAU_RAYLEIGH + tauMie() + TAU_OZONE;
    vec3 sunTransmitLow = exp(-tauZen * airmass(sunDir.y) * 1.0);

    /* --- Sky above the horizon --- */
    vec3 rdSky = normalize(vec3(rd.x, max(rd.y, 0.0015), rd.z));
    vec3 col = skyRadiance(rdSky, sunDir);
    col += sunRadiance(rdSky, sunDir, sunTransmit);

    float cloudCover;
    col = renderClouds(col, rdSky, sunDir, sunTransmitLow, cloudCover);

    /* --- Ocean below the horizon ---
     * Raymarched Gerstner height field lit entirely by the atmosphere above.
     * The mask band is a fraction of a degree wide; inside it the ocean's
     * own aerial perspective already matches the horizon sky, so the blend
     * is seamless. The coherent branch skips all water work for sky pixels. */
    float horizonMask = smoothstep(0.0018, -0.0018, rd.y);
    if (horizonMask > 0.001) {
        vec3 ro = vec3(0.0, CAM_HEIGHT, 0.0);
        vec3 oceanCol = shadeOcean(ro, rd, sunDir, sunTransmit, sunTransmitLow);
        col = mix(col, oceanCol, horizonMask);
    }

    /* --- In-camera glare (analytic bloom), partially blocked by clouds ---
     * The glare PSF is applied over the water too (it lives in the lens,
     * not the scene), just slightly damped below the horizon.              */
    float occl = mix(1.0, 0.22, cloudCover) * mix(1.0, 0.55, horizonMask);
    col += bloomApprox(rd, sunDir, sunTransmit, occl);

    return col;
}

/* ============================================================================
 *  MAIN
 * ==========================================================================*/

void main() {
    /* Aspect-corrected NDC, y in [-1, 1]. */
    vec2 ndc = (gl_FragCoord.xy * 2.0 - uResolution) / uResolution.y;

    /* Interactive cinematic camera: yaw about the world Y axis, pitch about
     * the camera right axis — no roll term exists, so the horizon always
     * stays level. Angles arrive pre-damped from the CPU orbit controller.
     * ~55 degrees vertical field of view.                                 */
    float cy = cos(uCamYaw),   sy = sin(uCamYaw);
    float cp = cos(uCamPitch), sp = sin(uCamPitch);
    vec3 fwd   = vec3(sy * cp, sp, cy * cp);
    vec3 right = vec3(cy, 0.0, -sy);
    vec3 up    = vec3(-sy * sp, cp, -cy * sp);
    float tanHalfFov = tan(radians(27.5));

    vec3 rd = normalize(fwd + tanHalfFov * (ndc.x * right + ndc.y * up));
    vec3 sunDir = sunDirection();

    /* ---- HDR scene ---- */
    vec3 hdr = renderScene(rd, sunDir);

    /* ---- Exposure adaptation ----
     * As the sun sinks the scene dims; a slow, slight exposure lift mimics
     * the eye (and an auto-exposure camera) adapting to dusk.             */
    float sunElevNorm = smoothstep(0.02, 0.06, sunDir.y);
    float exposure = mix(1.08, 0.90, sunElevNorm);
    hdr *= exposure;

    /* ---- Filmic tone mapping (HDR -> LDR) ---- */
    vec3 ldr = acesToneMap(hdr);

    /* ---- Very subtle chromatic aberration ----
     * True CA needs resampling; in a single pass we approximate a sub-pixel
     * radial channel shift with a first-order Taylor expansion using screen
     * derivatives. Strength grows with radius squared (corner-weighted).  */
    vec2 fromCenter = ndc * 0.5;
    vec2 caShiftPx = fromCenter * dot(fromCenter, fromCenter) * 1.8;
    vec3 dcx = dFdx(ldr);
    vec3 dcy = dFdy(ldr);
    ldr.r += dcx.r * caShiftPx.x + dcy.r * caShiftPx.y;
    ldr.b -= dcx.b * caShiftPx.x + dcy.b * caShiftPx.y;

    /* ---- Grade, vignette ---- */
    ldr = colorGrade(ldr);

    float vig = 1.0 - 0.30 * smoothstep(0.35, 1.55, length(ndc));
    ldr *= vig;

    /* ---- Gamma encode + dither ----
     * Ordered-ish screen hash dithering (+/- 0.5 LSB) defeats the banding
     * that smooth sky gradients otherwise show on 8-bit displays.         */
    ldr = pow(max(ldr, 0.0), vec3(1.0 / 2.2));
    float dither = (hash12(gl_FragCoord.xy + fract(uTime) * 61.7) - 0.5) / 255.0;
    ldr += dither;

    fragColor = vec4(ldr, 1.0);
}
